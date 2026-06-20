"""
L2 动态可塑性路由引擎
解决 F-02 拓扑固化断层
核心机制：
  1. 静态路由门控 + 推理时在线微调 (TTT)
  2. 弹性权重巩固 (EWC) 保护预训练核心拓扑
  3. 滑动窗口动量 (EMA) 抑制单步漂移
对应设计稿："推理时Hebbian自适应，让流形结构随输入自适应变形"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertBlock(nn.Module):
    """专家子网络：局域密集计算单元"""
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model * 2
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class EWCRegularizer:
    """
    弹性权重巩固 (Elastic Weight Consolidation)
    基于Fisher信息矩阵为预训练权重施加二次惩罚
    Loss_ewc = lambda/2 * sum F_i * (theta_i - theta*_i)^2
    """
    def __init__(self, lambda_ewc):
        self.lambda_ewc = lambda_ewc
        self.fisher = {}      # Fisher信息矩阵对角近似
        self.optimal = {}     # 预训练最优权重快照
        self.is_pretrained = False  # 预训练完成标志

    def finalize_pretraining(self, model, pretrain_data, num_samples=100):
        """
        预训练完成后调用此方法，计算Fisher并保存最优权重
        应在预训练阶段结束时调用，而非在线计算
        """
        print(f"[EWC] 预训练完成，计算Fisher矩阵 (samples={num_samples})")
        self.compute_fisher(model, pretrain_data, num_samples)
        self.is_pretrained = True

    def compute_fisher(self, model, data_loader, num_samples=100):
        """计算Fisher信息（简化版：在少量样本上估计梯度平方期望）"""
        for name, param in model.named_parameters():
            self.fisher[name] = torch.zeros_like(param)
        model.train()
        for i, x in enumerate(data_loader):
            if i >= num_samples:
                break
            model.zero_grad()
            out, _ = model(x, enable_ttt=False)  # 禁用TTT，仅计算前向
            loss = out.pow(2).mean()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.pow(2) / num_samples
        self._snapshot(model)

    def _snapshot(self, model):
        """记录预训练最优权重"""
        for name, param in model.named_parameters():
            self.optimal[name] = param.data.clone().detach()

    def penalty(self, model):
        # 未完成预训练时返回0，避免错误约束
        if not self.is_pretrained:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher and name in self.optimal:
                _loss = self.fisher[name] * (param - self.optimal[name]).pow(2)
                loss += _loss.sum()
        return self.lambda_ewc * loss / 2.0


class MomentumSmoother:
    """滑动窗口动量：EMA平滑TTT单步更新量"""
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.velocities = {}

    def smooth(self, name, grad):
        if name not in self.velocities:
            self.velocities[name] = torch.zeros_like(grad)
        self.velocities[name] = self.momentum * self.velocities[name] + (1 - self.momentum) * grad
        return self.velocities[name]

    def reset(self):
        self.velocities.clear()


class PlasticityMoELayer(nn.Module):
    """
    动态可塑性MoE层
    设计稿核心：局部重建误差驱动连接权重动态变化，模拟赫布学习
    """
    def __init__(self, num_experts, top_k, d_model,
                 plasticity_lr, weight_clip,
                 use_ewc, ewc_lambda,
                 use_ema, ema_momentum,
                 device="cpu"):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        self.plasticity_lr = plasticity_lr
        self.weight_clip = weight_clip
        self.use_ewc = use_ewc
        self.use_ema = use_ema
        self.device = device

        # 静态路由门控
        self.static_router = nn.Linear(d_model, num_experts).to(device)

        # 专家子网络集群
        self.experts = nn.ModuleList([
            ExpertBlock(d_model).to(device) for _ in range(num_experts)
        ])

        # 惯性制动：可塑性范围约束
        self.register_buffer("router_baseline", None)

        # EWC正则化器
        if self.use_ewc:
            self.ewc = EWCRegularizer(ewc_lambda)
        else:
            self.ewc = None

        # EMA动量平滑器
        if self.use_ema:
            self.ema = MomentumSmoother(ema_momentum)
        else:
            self.ema = None

        # 统计量
        self.update_history = []

    def forward(self, x, enable_ttt=True):
        """
        x: (batch, seq_len, d_model)
        enable_ttt: 是否启用推理时在线微调
        返回: (batch, seq_len, d_model), info_dict
        """
        batch, seq_len, d = x.shape

        # 1. 静态基础路由
        base_logits = self.static_router(x)  # (batch, seq_len, num_experts)
        base_weights = F.softmax(base_logits, dim=-1)

        # 2. 局部重建误差驱动可塑性调整（赫布学习模拟）
        adapted_weights = base_weights
        if enable_ttt and self.training:
            # 确保路由器参数需要梯度
            for param in self.static_router.parameters():
                param.requires_grad_(True)
            
            # 计算专家输出与输入的重建误差
            expert_outs = torch.stack([exp(x) for exp in self.experts], dim=2)  # (batch, seq_len, num_experts, d)
            # 加权组合
            weighted = (base_weights.unsqueeze(-1) * expert_outs).sum(dim=2)  # (batch, seq_len, d)
            recon_loss = F.mse_loss(weighted, x)

            # EWC惩罚（惯性制动：保护核心知识）
            if self.use_ewc and self.ewc is not None:
                recon_loss = recon_loss + self.ewc.penalty(self)

            # 反向传播更新static_router参数（真正的TTT在线学习）
            # 使用torch.autograd.grad只计算参数梯度，避免回传输入梯度
            self.static_router.zero_grad()
            params = list(self.static_router.parameters())
            grads = torch.autograd.grad(
                outputs=recon_loss,
                inputs=params,
                retain_graph=False,
                create_graph=False
            )

            with torch.no_grad():
                # 创建参数到梯度的映射
                param_grad_map = {param: grad for param, grad in zip(params, grads)}
                for name, param in self.static_router.named_parameters():
                    if param in param_grad_map and param_grad_map[param] is not None:
                        grad_update = param_grad_map[param]
                        # EMA平滑梯度
                        if self.use_ema:
                            grad_update = self.ema.smooth(name, grad_update)
                        # 单步梯度下降 + 裁剪
                        delta = self.plasticity_lr * grad_update
                        delta = torch.clamp(delta, -self.weight_clip, self.weight_clip)
                        param -= delta

            # 重新计算适应后的路由权重
            with torch.no_grad():
                adapted_logits = self.static_router(x)
                adapted_weights = F.softmax(adapted_logits, dim=-1)

            self.update_history.append({
                "recon_loss": recon_loss.item(),
                "weight_delta": (adapted_weights - base_weights).abs().mean().item()
            })

        # 3. Top-K路由选择（对称性破缺：仅激活k个专家）
        topk_vals, topk_idx = torch.topk(adapted_weights, self.top_k, dim=-1)
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

        # 4. 专家计算
        output = torch.zeros_like(x)
        for b in range(batch):
            for s in range(seq_len):
                for k in range(self.top_k):
                    expert_id = topk_idx[b, s, k].item()
                    weight = topk_vals[b, s, k]
                    output[b, s] += weight * self.experts[expert_id](x[b, s].unsqueeze(0)).squeeze(0)

        info = {
            "router_entropy": -(adapted_weights * (adapted_weights + 1e-9).log()).sum(dim=-1).mean().item(),
            "topk_idx": topk_idx.detach().cpu(),
            "adapted": enable_ttt and self.training
        }
        return output, info

    def ewc_penalty(self):
        if self.use_ewc and self.ewc is not None:
            return self.ewc.penalty(self)
        return torch.tensor(0.0, device=self.device)
