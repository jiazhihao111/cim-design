"""
使用Qwen模型验证存算一体架构效果 — 修复版
修复：
  1. 移除 set_attn_implementation('eager') — 使用默认sdpa模式
  2. 用注意力近似替代 output_attentions=True 的额外forward
  3. L1使用真实hidden states而非随机数据
  4. 添加CUDA事件真实性能测量
"""
import os
import time
import torch
import yaml
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import load_config, set_seed
from l1_micro_fusion import SRAMFusionEngine
from l2_plastic_router import PlasticityMoELayer


def estimate_attention_pattern(hidden_states):
    """
    从hidden states估计注意力模式（避免output_attentions=True的开销）
    使用hidden states的余弦相似度作为注意力近似
    hidden_states: (batch, seq_len, d_model)
    返回: (batch, seq_len, seq_len) 近似注意力权重
    """
    # 归一化
    h_norm = hidden_states / (hidden_states.norm(dim=-1, keepdim=True) + 1e-9)
    # 余弦相似度作为注意力近似
    sim = torch.bmm(h_norm, h_norm.transpose(-2, -1))  # (batch, seq, seq)
    # softmax归一化
    attn_approx = torch.softmax(sim / 0.1, dim=-1)  # 温度0.1使分布更集中
    return attn_approx


def load_qwen_model(model_path):
    """加载本地Qwen2-5-7B模型 — 不强制eager模式"""
    print(f"\n[1/3] 加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=True  # 8位压缩
    )
    # 修复：不移除sdpa模式，使用默认注意力实现
    # model.set_attn_implementation('eager') ← 删除这行
    
    model.eval()
    print(f"  模型加载完成，设备: {model.device}")
    return tokenizer, model


def benchmark_standard_inference(tokenizer, model, prompts, max_new_tokens=64):
    """标准推理基准测试"""
    print("\n[2/3] 标准推理基准测试")
    
    total_time = 0.0
    total_tokens = 0
    outputs = []
    
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 使用CUDA事件精准计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event) / 1000.0  # 秒
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        num_tokens = len(output[0]) - len(inputs["input_ids"][0])
        
        total_time += elapsed
        total_tokens += num_tokens
        outputs.append(response)
        
        print(f"  提示{i+1}: {prompt[:50]}...")
        print(f"    生成时间: {elapsed:.3f}s, {num_tokens} tokens, 速度: {num_tokens/elapsed:.1f} tok/s")
    
    avg_speed = total_tokens / total_time
    print(f"\n  标准推理平均速度: {avg_speed:.1f} tokens/s")
    
    return outputs, avg_speed, total_time


def benchmark_optimized_inference(tokenizer, model, prompts, cfg, max_new_tokens=64):
    """存算一体优化推理测试 — 修复版：无额外attention forward"""
    print("\n[3/3] 存算一体优化推理测试")
    
    l1c = cfg["l1_fusion"]
    l2c = cfg["l2_router"]
    dtype = torch.bfloat16
    qwen_hidden_dim = model.config.hidden_size
    
    # 初始化L1融合引擎
    fusion_engine = SRAMFusionEngine(
        sram_capacity=l1c["sram_capacity"],
        block_size=l1c["block_size"],
        d_model=l1c["d_model"],
        entropy_threshold_high=l1c["entropy_threshold_high"],
        entropy_threshold_low=l1c["entropy_threshold_low"],
        async_prefetch_window=l1c["async_prefetch_window"],
        flash_attention_sim=True
    ).to(model.device).to(dtype)
    
    router = PlasticityMoELayer(
        num_experts=l2c["num_experts"],
        top_k=l2c["top_k"],
        d_model=qwen_hidden_dim,
        plasticity_lr=l2c["plasticity_lr"],
        weight_clip=l2c["weight_clip"],
        use_ewc=l2c["use_ewc"],
        ewc_lambda=l2c["ewc_lambda"],
        use_ema=l2c["use_ema"],
        ema_momentum=l2c["ema_momentum"]
    ).to(model.device).to(dtype)
    
    router.eval()
    
    total_time = 0.0
    total_tokens = 0
    outputs = []
    sram_stats = []
    
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        with torch.no_grad():
            # 修复：先获取embedding（不需要额外forward）
            hidden_states = model.model.embed_tokens(inputs["input_ids"])
            
            # 修复：用hidden states估计注意力（无需output_attentions=True）
            attn_approx = estimate_attention_pattern(hidden_states)
            
            # 修复：L1使用真实hidden states（非随机数据）
            # 将d_model=128的L1对接到真实3584维hidden states
            l1_input = hidden_states[:, :, :l1c["d_model"]]
            _, l1_stats, levels = fusion_engine(l1_input, attn_approx)
            sram_stats.append(l1_stats)

            # === P3: 通过forward hook将稀疏mask注入模型attention ===
            seq_len = inputs["input_ids"].shape[1]
            sparse_mask = torch.ones(1, 1, seq_len, seq_len, device=model.device, dtype=dtype)
            tail_positions = (levels[0] == 0).nonzero(as_tuple=True)[0]
            if len(tail_positions) > 0:
                sparse_mask[:, :, tail_positions, :] = 0.0

            def make_sparse_hook(sm):
                def hook(module, args, kwargs):
                    am = kwargs.get('attention_mask', None)
                    if am is not None and am.dim() == 4:
                        # 仅在prefill阶段（形状匹配时）应用稀疏mask
                        if am.shape[-1] == sm.shape[-1] and am.shape[-2] == sm.shape[-2]:
                            kwargs['attention_mask'] = am & sm
                        # 解码阶段形状不匹配，跳过稀疏mask
                    return args, kwargs
                return hook

            hooks = [layer.self_attn.register_forward_pre_hook(
                make_sparse_hook(sparse_mask), with_kwargs=True)
                for layer in model.model.layers]

            # 标准的generate推理（带稀疏注意力mask）
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

            for h in hooks:
                h.remove()
        
        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event) / 1000.0
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        num_tokens = len(output[0]) - len(inputs["input_ids"][0])
        
        total_time += elapsed
        total_tokens += num_tokens
        outputs.append(response)
        
        sram_hit_rate = l1_stats["sram_hits"] / max(l1_stats["sram_hits"] + l1_stats["sram_misses"], 1)
        print(f"  提示{i+1}: {prompt[:50]}...")
        print(f"    生成时间: {elapsed:.3f}s, {num_tokens} tokens, 速度: {num_tokens/elapsed:.1f} tok/s")
        print(f"    SRAM命中率: {sram_hit_rate*100:.1f}%, 融合耗时: {l1_stats['fusion_time_ms']:.2f}ms")
    
    avg_speed = total_tokens / total_time
    avg_sram_hit = np.mean([s["sram_hits"] / max(s["sram_hits"] + s["sram_misses"], 1) for s in sram_stats])
    
    print(f"\n  存算一体优化平均速度: {avg_speed:.1f} tokens/s")
    print(f"  平均SRAM命中率: {avg_sram_hit*100:.1f}%")
    
    return outputs, avg_speed, total_time, avg_sram_hit


def main():
    model_path = r"C:\Users\51615\.cache\modelscope\Qwen2___5-7B-Instruct"
    cfg = load_config("config.yaml")
    set_seed(cfg["system"]["seed"])
    
    print("=" * 70)
    print("Qwen2-5-7B-Instruct 存算一体架构效果验证 (修复版)")
    print("=" * 70)
    
    tokenizer, model = load_qwen_model(model_path)
    
    prompts = [
        "解释什么是存算一体架构？",
        "用简单的语言解释量子计算的基本原理",
    ]
    
    # 标准推理
    std_outputs, std_speed, std_time = benchmark_standard_inference(tokenizer, model, prompts)
    
    # 存算一体优化（修复版：无额外attention forward）
    opt_outputs, opt_speed, opt_time, opt_sram_hit = benchmark_optimized_inference(
        tokenizer, model, prompts, cfg
    )
    
    # 结果对比
    print("\n" + "=" * 70)
    print("性能对比汇总")
    print("=" * 70)
    print(f"{'方法':<20} {'速度(tok/s)':<15} {'耗时(s)':<15} {'SRAM命中率':<15}")
    print(f"{'标准推理':<20} {std_speed:<15.1f} {std_time:<15.3f} {'-':<15}")
    print(f"{'存算一体优化':<20} {opt_speed:<15.1f} {opt_time:<15.3f} {f'{opt_sram_hit*100:.1f}%':<15}")
    print("=" * 70)
    
    speed_diff = std_speed - opt_speed
    if speed_diff > 0:
        print(f"\n⚠️ 存算一体优化比标准推理慢 {speed_diff:.1f} tok/s")
        print("  原因分析:")
        print("  1. L1/L2模块在PyTorch中作为额外层运行，增加了计算图")
        print("  2. 真正的存算一体优势需要CUDA kernel级别的算子融合")
        print("  3. 当前软件模拟层无法绕过冯·诺依曼瓶颈")
    else:
        speedup = -speed_diff
        print(f"\n✅ 存算一体优化比标准推理快 {speedup:.1f} tok/s ({speedup/std_speed*100:.1f}%)")


if __name__ == "__main__":
    main()
