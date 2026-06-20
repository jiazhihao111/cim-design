"""
跨尺度反馈环路
解决 F-04 跨尺度无反馈断层
核心机制：
  1. 宏观弛豫稳态 -> 中观路由门控（动态优化专家激活）
  2. 中观路由语义 -> 微观缓存调度（动态调整SRAM锚点优先级）
对应设计稿："自上而下的粗粒化调控，强化时空耦合与高阶涌现"
"""
import torch
import torch.nn as nn


class CrossScaleFeedback(nn.Module):
    """
    跨尺度反向反馈通道
    连接 L3(宏观) -> L2(中观) -> L1(微观)
    """
    def __init__(self, d_model, num_experts,
                 macro_to_meso_scale,
                 macro_to_micro_scale,
                 meso_to_micro_scale,
                 device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.scale_m2meso = macro_to_meso_scale
        self.scale_m2micro = macro_to_micro_scale
        self.scale_meso2micro = meso_to_micro_scale

        # 宏观语义 -> 中观路由偏好投影
        self.macro_to_router = nn.Linear(d_model, num_experts).to(device)

        # 宏观语义 -> 微观熵调节
        self.macro_to_entropy = nn.Linear(d_model, 1).to(device)

        # 中观路由 -> 微观缓存优先级
        self.router_to_cache = nn.Linear(num_experts, d_model).to(device)

    def forward(self, macro_state, meso_router_weights, micro_levels):
        """
        macro_state: (batch, seq_len, d_model)  L3稳态输出
        meso_router_weights: (batch, seq_len, num_experts) L2路由权重
        micro_levels: (batch, seq_len) L1缓存分级 (0=长尾,1=普通,2=核心)
        返回: 反馈调制后的各层控制信号
        """
        # 1. 宏观 -> 中观：语义驱动的路由偏置
        router_bias = self.macro_to_router(macro_state)  # (batch, seq_len, num_experts)
        modulated_router = meso_router_weights + self.scale_m2meso * torch.tanh(router_bias)
        modulated_router = torch.softmax(modulated_router, dim=-1)

        # 2. 宏观 -> 微观：语义重要性动态调节熵阈值
        importance = torch.sigmoid(self.macro_to_entropy(macro_state)).squeeze(-1)  # (batch, seq_len)
        # importance高 -> 提升为core，importance低 -> 降级为tail
        adjusted_levels = micro_levels.clone().float()
        adjusted_levels = adjusted_levels + self.scale_m2micro * (importance * 2 - 1)
        adjusted_levels = torch.clamp(adjusted_levels, 0, 2).round().long()

        # 3. 中观 -> 微观：路由激活模式影响缓存驻留策略
        cache_priority = self.router_to_cache(meso_router_weights)  # (batch, seq_len, d_model)
        # 取平均作为位置重要性代理
        cache_priority_score = cache_priority.mean(dim=-1)  # (batch, seq_len)

        feedback_info = {
            "router_bias_norm": router_bias.norm(dim=-1).mean().item(),
            "importance_mean": importance.mean().item(),
            "cache_priority_mean": cache_priority_score.mean().item(),
            "level_change_rate": (adjusted_levels != micro_levels).float().mean().item()
        }
        return modulated_router, adjusted_levels, cache_priority_score, feedback_info
