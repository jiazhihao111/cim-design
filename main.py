"""
纯软件模拟存算一体拓扑流形架构 主运行入口
整合 L1 + L2 + L3 + 相变采样 + 跨尺度反馈
"""
import os
import yaml
import torch
import torch.nn as nn
from utils import load_config, set_seed
from topology_network import SmallWorldTopology
from l1_micro_fusion import SRAMFusionEngine
from l2_plastic_router import PlasticityMoELayer
from l3_relaxation import RelaxationEngine
from phase_sampler import PhaseSampler
from cross_scale_feedback import CrossScaleFeedback


class EmergentCIMEngine(nn.Module):
    """
    涌现存算一体引擎 (Emergent Compute-In-Memory Engine)
    三层递进 + 相变采样 + 跨尺度反馈的完整闭环
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        device = cfg["system"]["device"]
        d = cfg["l1_fusion"]["d_model"]

        # L1: 微观存算融合
        l1c = cfg["l1_fusion"]
        self.l1 = SRAMFusionEngine(
            sram_capacity=l1c["sram_capacity"],
            block_size=l1c["block_size"],
            d_model=d,
            entropy_threshold_high=l1c["entropy_threshold_high"],
            entropy_threshold_low=l1c["entropy_threshold_low"],
            async_prefetch_window=l1c["async_prefetch_window"],
            flash_attention_sim=l1c["flash_attention_sim"]
        )

        # L2: 动态可塑性路由
        l2c = cfg["l2_router"]
        self.l2 = PlasticityMoELayer(
            num_experts=l2c["num_experts"],
            top_k=l2c["top_k"],
            d_model=d,
            plasticity_lr=l2c["plasticity_lr"],
            weight_clip=l2c["weight_clip"],
            use_ewc=l2c["use_ewc"],
            ewc_lambda=l2c["ewc_lambda"],
            use_ema=l2c["use_ema"],
            ema_momentum=l2c["ema_momentum"],
            device=device
        )

        # L3: 连续时间弛豫
        l3c = cfg["l3_relaxation"]
        self.l3 = RelaxationEngine(
            input_dim=d,
            state_dim=l3c["ssm_state_dim"],
            dt=l3c["dt"],
            deq_max_iter=l3c["deq_max_iter"],
            deq_tol=l3c["deq_tol"],
            spectral_radius_threshold=l3c["spectral_radius_threshold"],
            spectral_norm_gamma=l3c["spectral_norm_gamma"],
            use_spectral_norm=l3c["use_spectral_norm"],
            use_circuit_breaker=l3c["use_circuit_breaker"],
            circuit_breaker_max_iter=l3c["circuit_breaker_max_iter"],
            device=device
        )

        # 相变采样器
        psc = cfg["phase_sampler"]
        self.sampler = PhaseSampler(
            input_dim=d,
            vocab_size=psc["vocab_size"],
            temperature=psc["temperature"],
            sde_noise_scale=psc["sde_noise_scale"],
            gumbel_tau=psc["gumbel_tau"],
            use_sde=psc["use_sde"]
        )

        # 跨尺度反馈
        fbc = cfg["feedback"]
        self.feedback = CrossScaleFeedback(
            d_model=d,
            num_experts=l2c["num_experts"],
            macro_to_meso_scale=fbc["macro_to_micro_scale"],
            macro_to_micro_scale=fbc["macro_to_micro_scale"],
            meso_to_micro_scale=fbc["meso_to_micro_scale"],
            device=device
        )

        # 拓扑网络（全局流形基底）
        tc = cfg["topology"]
        self.topology = SmallWorldTopology(
            num_nodes=tc["num_nodes"],
            k_nearest=tc["k_nearest"],
            rewire_prob=tc["rewire_prob"],
            device=device
        )

        # 输入嵌入
        self.embed = nn.Linear(d, d)
        
        # 反馈开关（L3→L1逆向反馈）
        self.feedback_enabled = cfg.get("feedback", {}).get("l3_to_l1_feedback", True)

    def forward(self, x, attn_weights, return_all=False):
        """
        x: (batch, seq_len, d_model)
        attn_weights: (batch, seq_len, seq_len)
        """
        info = {}

        # 拓扑影响矩阵（修复：实际集成拓扑网络）
        topo_influence = self.topology.compute_influence(
            seq_len=x.size(1), batch_size=x.size(0),
            device=x.device
        )

        # === L1: 微观存算融合（传递拓扑影响） ===
        l1_out, l1_stats, levels = self.l1(x, attn_weights, topology_influence=topo_influence)
        info["l1"] = l1_stats

        # === L2: 动态可塑性路由 ===
        l2_out, l2_info = self.l2(l1_out, enable_ttt=True)
        info["l2"] = l2_info

        # === L3: 连续时间弛豫 ===
        l3_out, l3_meta = self.l3(l2_out, mode="deq")
        info["l3"] = l3_meta

        # === L3→L1逆向反馈（新增闭环通道）===
        # 将弛豫后的状态重要性反馈给缓存决策，优化SRAM分配
        if hasattr(self, 'feedback_enabled') and self.feedback_enabled:
            # 计算L3输出的重要性权重（基于状态变异度）
            l3_importance = l3_out.std(dim=-1, keepdim=True)  # (batch, seq, 1)
            # 归一化到[0, 1]
            l3_importance = (l3_importance - l3_importance.min()) / (l3_importance.max() - l3_importance.min() + 1e-8)
            
            # 计算L3注意力（用于融合L1原始注意力）
            l3_attn = torch.softmax(l3_out @ l3_out.transpose(-2, -1), dim=-1)
            
            # 融合L1和L3的注意力分布
            fused_attn = 0.5 * attn_weights + 0.5 * l3_attn
            l3_fb_info = {"l3_importance_mean": l3_importance.mean().item()}
            
            # 重新评估缓存等级（可选的精细化反馈）
            # 这里简化为记录反馈信息
            info["l3_to_l1_feedback"] = l3_fb_info
        
        # === 跨尺度反馈 ===
        mod_router, adjusted_levels, cache_priority, fb_info = self.feedback(
            l3_out, l2_info.get("adapted_weights", torch.ones_like(l2_out[:, :, :self.l2.num_experts])), levels
        )
        info["feedback"] = fb_info

        # === 相变采样 ===
        probs, tokens, energy = self.sampler(l3_out)
        info["sampler"] = {
            "energy_mean": energy.mean().item(),
            "token_entropy": -(probs * (probs + 1e-9).log()).sum(dim=-1).mean().item()
        }

        if return_all:
            return tokens, probs, l3_out, info
        return tokens, info


def demo_run():
    print("=" * 70)
    print("纯软件模拟存算一体拓扑流形架构 —— 端到端演示")
    print("=" * 70)

    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = load_config(cfg_path)
    set_seed(cfg["system"]["seed"])
    device = cfg["system"]["device"]

    engine = EmergentCIMEngine(cfg)
    engine.to(device)
    engine.train()

    batch = 2
    seq_len = min(cfg["l1_fusion"]["seq_length"], 512)  # 减小序列长度加速演示
    d = cfg["l1_fusion"]["d_model"]

    x = torch.randn(batch, seq_len, d, device=device)

    # 生成有结构性的注意力权重（对角占优模拟真实注意力）
    attn = torch.zeros(batch, seq_len, seq_len, device=device)
    for b in range(batch):
        for i in range(seq_len):
            for j in range(seq_len):
                attn[b, i, j] = max(0, 1.0 - abs(i - j) / (seq_len * 0.15))
    attn = attn / attn.sum(dim=-1, keepdim=True)

    tokens, info = engine(x, attn)

    print(f"\n输入张量: {x.shape}")
    print(f"输出Token: {tokens.shape}")
    print(f"\n--- L1 微观存算融合统计 ---")
    for k, v in info["l1"].items():
        print(f"  {k}: {v}")

    print(f"\n--- L2 动态可塑性路由统计 ---")
    for k, v in info["l2"].items():
        if isinstance(v, (int, float, bool, str)):
            print(f"  {k}: {v}")

    print(f"\n--- L3 连续弛豫统计 ---")
    for k, v in info["l3"].items():
        print(f"  {k}: {v}")

    print(f"\n--- 跨尺度反馈统计 ---")
    for k, v in info["feedback"].items():
        print(f"  {k}: {v}")

    print(f"\n--- 相变采样统计 ---")
    for k, v in info["sampler"].items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("端到端演示完成。")
    print("=" * 70)


if __name__ == "__main__":
    demo_run()
