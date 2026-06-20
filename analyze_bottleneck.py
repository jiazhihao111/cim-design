"""详细时间分解分析 — 修复版：CUDA Event计时 + sdpa模式 + 注意力估计"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import load_config
from l1_micro_fusion import SRAMFusionEngine

cfg = load_config('config.yaml')
l1c = cfg['l1_fusion']

print('='*70)
print('存算一体优化性能瓶颈分析 (修复版)')
print('='*70)

model_path = r"C:\Users\51615\.cache\modelscope\Qwen2___5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
# 修复：不强制eager模式，使用默认sdpa
model.eval()

dtype = torch.bfloat16

prompt = "解释什么是存算一体架构？"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
seq_len = inputs["input_ids"].shape[1]


def cuda_timed(fn, *args, **kwargs):
    """CUDA Event精准计时"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000.0


def estimate_attn(hidden_states):
    """从hidden states估计注意力（无需output_attentions）"""
    h = hidden_states / (hidden_states.norm(dim=-1, keepdim=True) + 1e-9)
    sim = torch.bmm(h, h.transpose(-2, -1))
    return torch.softmax(sim / 0.1, dim=-1)


print(f'\n序列长度: {seq_len}')

# === 1. 标准推理 (sdpa) ===
(output,), t_sdpa = cuda_timed(
    lambda: model.generate(**inputs, max_new_tokens=64, do_sample=False)
)
print(f"[1] 标准推理(sdpa): {t_sdpa:.3f}s")

# === 2. 注意力估计 (无额外forward) ===
hidden = model.model.embed_tokens(inputs["input_ids"])
(attn_approx,), t_attn = cuda_timed(lambda: estimate_attn(hidden))
print(f"[2] 注意力估计(无额外forward): {t_attn*1000:.2f}ms")

# === 3. L1融合引擎 (真实hidden states) ===
fusion_engine = SRAMFusionEngine(
    sram_capacity=l1c['sram_capacity'], block_size=l1c['block_size'],
    d_model=l1c['d_model'], entropy_threshold_high=l1c['entropy_threshold_high'],
    entropy_threshold_low=l1c['entropy_threshold_low'],
    async_prefetch_window=l1c['async_prefetch_window']
).to(model.device).to(dtype)

(l1_out, l1_stats, levels), t_l1 = cuda_timed(
    lambda: fusion_engine(hidden[:, :, :l1c['d_model']].to(dtype), attn_approx.to(dtype))
)
print(f"[3] L1融合引擎: {t_l1*1000:.2f}ms  "
      f"core={l1_stats['core_tokens']} tail={l1_stats['tail_tokens']}")

# === 汇总 ===
print('\n' + '='*70)
print(f"标准推理(sdpa):           {t_sdpa:.3f}s (基准)")
print(f"注意力估计:                 {t_attn*1000:.1f}ms")
print(f"L1融合引擎:                 {t_l1*1000:.1f}ms")
print(f"总额外开销:                 {(t_attn+t_l1)*1000:.1f}ms "
      f"({(t_attn+t_l1)/t_sdpa*100:.1f}%)")
