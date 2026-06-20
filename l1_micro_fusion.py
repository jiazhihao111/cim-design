"""
L1 微观存算融合引擎 — 修复版
问题修复：
  1. 替换随机投影(torch.randn)为真正的注意力加权聚合
  2. 修复entropy/concentration双重计算(逻辑反转) 
  3. 添加topology网络集成接口
  4. 添加CUDA事件真实计时替代理论公式
"""
import torch
import torch.nn as nn
import numpy as np


class EntropyEvaluator(nn.Module):
    """注意力熵评估器：量化token语义重要性"""
    def __init__(self, entropy_threshold_high, entropy_threshold_low):
        super().__init__()
        self.high_thr = entropy_threshold_high  # 集中度阈值 → 核心
        self.low_thr = entropy_threshold_low    # 集中度阈值 → 长尾

    def forward(self, attn_weights):
        """
        attn_weights: (batch, seq_len, seq_len)
        返回: 每个位置的重要性等级 (batch, seq_len)
              2=核心锚点, 1=普通, 0=长尾
        """
        eps = 1e-9
        attn = attn_weights.clamp(min=eps)
        seq_len = attn.size(-1)
        
        # 注意力集中度 = 1 - 归一化熵
        # 集中度越高 → token越核心（注意力集中在少数位置）
        entropy = -(attn * attn.log()).sum(dim=-1)  # (batch, seq_len)
        max_entropy = np.log(seq_len)
        concentration = 1.0 - (entropy / max_entropy)  # 归一化集中度
        
        # 分级
        levels = torch.zeros_like(concentration, dtype=torch.long)
        levels[concentration >= self.high_thr] = 2   # 核心锚点
        levels[(concentration >= self.low_thr) & (concentration < self.high_thr)] = 1  # 普通
        levels[concentration < self.low_thr] = 0      # 长尾
        
        return levels, concentration


class SRAMFusionEngine(nn.Module):
    """
    模拟片上SRAM存算融合 — 修复版
    用真正的注意力加权聚合代替随机投影
    """
    def __init__(self, sram_capacity, block_size, d_model,
                 entropy_threshold_high, entropy_threshold_low,
                 async_prefetch_window, flash_attention_sim=True):
        super().__init__()
        self.sram_capacity = sram_capacity
        self.block_size = block_size
        self.d_model = d_model
        self.flash_sim = flash_attention_sim
        self.prefetch_window = async_prefetch_window

        self.entropy_eval = EntropyEvaluator(entropy_threshold_high, entropy_threshold_low)

        # 注意力加权输出投影（替代随机投影）
        self.attn_proj = nn.Linear(d_model, d_model)
        
        # 存算一体"原位计算"权重（模拟SRAM内计算）
        # 真实存算一体中，权重存储在SRAM中，计算在SRAM旁完成
        self.sram_weight = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        
        # CUDA事件计时器
        self.cuda_events = {"fusion_ms": 0.0, "entropy_ms": 0.0}

    def forward(self, x, attn_weights, topology_influence=None):
        """
        真正的注意力加权融合（非随机投影）
        x: (batch, seq_len, d_model)
        attn_weights: (batch, seq_len, seq_len)
        topology_influence: (batch, seq_len, seq_len) 或 None — 拓扑网络影响
        """
        batch, seq_len, d = x.shape
        
        # CUDA事件计时
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # 1. 熵评估，分级
        levels, concentration = self.entropy_eval(attn_weights)
        
        # 2. 注意力加权聚合（替代随机投影）
        # 使用注意力权重对value进行聚合: out = attn @ x
        # 这是真正的"注意力融合"操作
        if attn_weights.dim() == 3:
            # attn_weights: (batch, seq_len, seq_len)
            # x: (batch, seq_len, d)
            attn_out = torch.bmm(attn_weights, x)  # (batch, seq_len, d)
        else:
            attn_out = x
        
        # 3. 应用SRAM存储权重（模拟片上计算）
        fused_out = self.attn_proj(attn_out)  # 先投影到标准空间
        
        # 4. 拓扑影响注入：如果提供了拓扑网络的连接矩阵
        if topology_influence is not None:
            # topology_influence: (batch, seq_len, seq_len) 连接强度
            topo_out = torch.bmm(topology_influence, fused_out)
            fused_out = fused_out + 0.1 * topo_out  # 拓扑作为弱先验
        
        # 5. SRAM驻留模拟 + 访存统计
        core_mask = (levels == 2).float()
        tail_mask = (levels == 0).float()
        core_tokens = core_mask.sum().item()
        tail_tokens = tail_mask.sum().item()
        
        sram_needed = core_tokens * d
        sram_hits = core_tokens if sram_needed <= self.sram_capacity else 0
        sram_misses = core_tokens - sram_hits
        
        # 6. 长尾异步预取（模拟）
        hbm_read = tail_tokens * d * 4  # float32
        hbm_write = seq_len * d * 4
        
        end.record()
        torch.cuda.synchronize()
        fusion_ms = start.elapsed_time(end)
        self.cuda_events["fusion_ms"] = fusion_ms

        stats = {
            "core_tokens": int(core_tokens),
            "tail_tokens": int(tail_tokens),
            "hbm_read_bytes": int(hbm_read),
            "hbm_write_bytes": int(hbm_write),
            "sram_hits": int(sram_hits),
            "sram_misses": int(sram_misses),
            "core_ratio": float(core_mask.mean()),
            "fusion_time_ms": fusion_ms,
        }
        return fused_out, stats, levels


class KVPruner(nn.Module):
    """实际的KV Cache压缩器 — 基于重要性分级剪枝长尾token"""
    def __init__(self, keep_ratio=0.3):
        super().__init__()
        self.keep_ratio = keep_ratio

    def prune(self, levels, k_cache, v_cache):
        """
        levels: (batch, seq_len) — L1分级结果
        k_cache, v_cache: (batch, seq_len, num_heads, head_dim)
        返回: 压缩后的k/v (减少seq_len维度)
        """
        batch, seq_len = levels.shape
        core_mask = (levels == 2)    # 核心token
        normal_mask = (levels == 1)   # 普通token
        tail_mask = (levels == 0)     # 长尾token

        # 核心+普通全部保留，长尾按keep_ratio随机采样
        tail_indices = torch.where(tail_mask[0])[0]
        n_keep_tail = max(1, int(len(tail_indices) * self.keep_ratio))
        keep_tail = tail_indices[torch.randperm(len(tail_indices))[:n_keep_tail]]

        # 构造保留索引
        keep_mask = core_mask | normal_mask
        keep_mask[0, keep_tail] = True

        # 实际剪枝
        pruned_k = k_cache[:, keep_mask[0], :, :]
        pruned_v = v_cache[:, keep_mask[0], :, :]

        compression_ratio = pruned_k.shape[1] / seq_len
        return pruned_k, pruned_v, keep_mask, compression_ratio
