"""
L3 连续时间弛豫引擎
解决 F-03 弛豫发散死锁断层
核心机制：
  1. SSM状态空间模型: dh/dt = A h + B x
  2. DEQ隐式不动点求解: h* = f(h*, x)，Anderson加速
  3. 动态谱归一化: 实时约束 rho(A) < 1
  4. 断路器: 超时降级为离散SSM步进
对应设计稿："连续弛豫实现流形稳态收敛"
"""
import logging
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # 默认只显示WARNING及以上级别


class SSMCell(nn.Module):
    """
    结构化状态空间模型 (Simplified S5/S4 flavor)
    连续时间动力学: dh/dt = A h + B x
    离散化: h_{t+1} = (I + dt A) h_t + dt B x_t
    """
    def __init__(self, input_dim, state_dim, dt):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.dt = dt

        # 状态转移矩阵 A (可学习，但需保证稳定性)
        # 稳定初始化：使用对角矩阵 + 小扰动，确保谱半径 < 1
        # 这样 effective_A = I + dt*A 的谱半径在0.9左右
        diag_val = -0.5 / dt  # 使谱半径约为0.5，dt=0.1时谱半径≈0.95
        self.A = nn.Parameter(torch.eye(state_dim) * diag_val + torch.randn(state_dim, state_dim) * 0.01)
        # 输入投影 B
        self.B = nn.Linear(input_dim, state_dim, bias=False)
        # 输出投影 C
        self.C = nn.Linear(state_dim, input_dim, bias=False)
        # 跳跃连接 D
        self.D = nn.Linear(input_dim, input_dim, bias=False)

    def effective_A(self):
        """离散化后的有效状态转移矩阵"""
        I = torch.eye(self.state_dim, device=self.A.device, dtype=self.A.dtype)
        return I + self.dt * self.A

    def spectral_radius(self):
        """计算谱半径 rho(A)"""
        eff_A = self.effective_A()
        eigs = torch.linalg.eigvals(eff_A)
        return eigs.abs().max().item()

    def forward_step(self, h, x):
        """单步前向: h_{t+1} = eff_A @ h + dt * B(x)"""
        eff_A = self.effective_A()
        h_next = torch.matmul(h, eff_A.t()) + self.dt * self.B(x)
        y = self.C(h_next) + self.D(x)
        return h_next, y

    def forward_sequence(self, x, h0=None):
        """
        序列前向: x (batch, seq_len, input_dim)
        返回: y (batch, seq_len, input_dim), 最终状态 h
        """
        batch, seq_len, _ = x.shape
        if h0 is None:
            h = torch.zeros(batch, self.state_dim, device=x.device, dtype=x.dtype)
        else:
            h = h0
        outputs = []
        for t in range(seq_len):
            h, y = self.forward_step(h, x[:, t, :])
            outputs.append(y)
        return torch.stack(outputs, dim=1), h


class SpectralNormalizer:
    """动态谱归一化：当谱半径趋近临界值时收缩权重"""
    def __init__(self, threshold, gamma):
        self.threshold = threshold
        self.gamma = gamma

    def normalize(self, ssm: SSMCell):
        rho = ssm.spectral_radius()
        if rho > self.threshold:
            scale = self.gamma / rho
            with torch.no_grad():
                ssm.A.data *= scale
            return True, rho, scale
        return False, rho, 1.0


class DEQSolver:
    """
    隐式层不动点求解器
    求解 h* = f(h*, x) 通过 Anderson 加速 + 谱归一化 + 断路器
    """
    def __init__(self, ssm: SSMCell, max_iter, tol,
                 spectral_normalizer: SpectralNormalizer,
                 use_circuit_breaker=True, circuit_breaker_max_iter=30):
        self.ssm = ssm
        self.max_iter = max_iter
        self.tol = tol
        self.spectral_norm = spectral_normalizer
        self.use_cb = use_circuit_breaker
        self.cb_iter = circuit_breaker_max_iter
        # 定义安全态（零向量）
        self._safe_state_shape = (ssm.state_dim,)
        self._last_failure_state = None

    def get_safe_state(self, batch_size, device, dtype):
        """返回安全态（零向量）"""
        return torch.zeros(batch_size, *self._safe_state_shape, device=device, dtype=dtype)

    def solve(self, x, h_init=None):
        """
        求解不动点 h* = f(h*, x)
        x: (batch, input_dim)
        返回: h_star, converge_flag, num_iter, meta_info
        """
        batch = x.size(0)
        if h_init is None:
            h = torch.zeros(batch, self.ssm.state_dim, device=x.device, dtype=x.dtype)
        else:
            h = h_init

        # Anderson加速历史窗口
        m = 3
        history = []
        residuals = []

        converged = False
        fallback_triggered = False  # 初始化fallback标志
        final_iter = self.max_iter

        # 统计信息
        spectral_trigger_count = 0  # 谱归一化触发次数
        steps_to_converge = []      # 记录每步收敛所需迭代次数
        
        for k in range(self.max_iter):
            # 动态谱归一化（惯性制动注入）
            triggered, rho, scale = self.spectral_norm.normalize(self.ssm)
            
            # 记录谱归一化触发
            if triggered:
                spectral_trigger_count += 1
                logger.debug(f"  [Step {k}] 谱归一化触发: ρ={rho:.6f} > {self.spectral_norm.threshold}, scale={scale:.6f}")

            # 不动点迭代步
            h_next, _ = self.ssm.forward_step(h, x)
            residual = (h_next - h).norm(dim=-1).max().item()

            history.append(h_next)
            residuals.append(residual)

            # Anderson加速（简化版：残差外推）
            if len(history) >= 2:
                # 极简Anderson：使用最后两步做线性外推
                if len(history) >= m + 1:
                    # 清空旧历史保持窗口
                    history = history[-m:]
                    residuals = residuals[-m:]
                # 这里简化不做完整最小二乘，仅做动量混合
                alpha = 0.3
                if len(history) >= 2:
                    h_next = (1 + alpha) * h_next - alpha * history[-2]

            h = h_next

            # 收敛检测
            if residual < self.tol:
                converged = True
                final_iter = k + 1
                steps_to_converge.append(k + 1)
                logger.debug(f"  [收敛成功] 步数={k+1}, 最终残差={residual:.8f}, 谱半径={rho:.6f}, 谱归一化触发={spectral_trigger_count}次")
                break

            # 断路器（死锁逃生）- 返回安全态而非发散状态
            if self.use_cb and k >= self.cb_iter:
                logger.debug(f"  [CircuitBreaker 触发]")
                logger.debug(f"    ├── 触发步数: {k+1} (超过阈值{self.cb_iter})")
                logger.debug(f"    ├── 最终残差: {residual:.8f} (未达到容差{self.tol:.2e})")
                logger.debug(f"    ├── 谱半径ρ: {rho:.6f}")
                logger.debug(f"    ├── 谱归一化触发次数: {spectral_trigger_count}")
                logger.debug(f"    └── 原因: max_iter_exceeded")
                # 保存故障上下文
                self._last_failure_state = h.clone()
                fallback_triggered = True
                # 返回安全态（零向量）而非不稳定状态
                safe_h = self.get_safe_state(batch, x.device, x.dtype)
                meta = {
                    "converged": False,
                    "iterations": self.cb_iter,
                    "final_residual": residual,
                    "spectral_triggered": triggered,
                    "final_rho": rho,
                    "circuit_breaker_triggered": True,
                    "circuit_breaker_reason": "max_iter_exceeded",
                    "scale": scale,
                    "spectral_trigger_count": spectral_trigger_count,
                    "steps_to_converge": steps_to_converge
                }
                return safe_h, False, self.cb_iter, meta

        meta = {
            "converged": converged,
            "iterations": final_iter,
            "final_residual": residual,
            "spectral_triggered": triggered,
            "final_rho": rho,
            "fallback_triggered": fallback_triggered,
            "scale": scale,
            "spectral_trigger_count": spectral_trigger_count,
            "steps_to_converge": steps_to_converge
        }
        return h, converged, final_iter, meta


class RelaxationEngine(nn.Module):
    """L3宏观弛豫引擎：封装SSM+DEQ+谱控制+断路器"""
    def __init__(self, input_dim, state_dim, dt,
                 deq_max_iter, deq_tol,
                 spectral_radius_threshold, spectral_norm_gamma,
                 use_spectral_norm, use_circuit_breaker, circuit_breaker_max_iter,
                 device="cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.device = device

        self.ssm = SSMCell(input_dim, state_dim, dt).to(device)
        self.spectral_norm = SpectralNormalizer(spectral_radius_threshold, spectral_norm_gamma)
        self.deq_solver = DEQSolver(
            self.ssm, deq_max_iter, deq_tol,
            self.spectral_norm,
            use_circuit_breaker, circuit_breaker_max_iter
        )

    def forward(self, x, mode="deq"):
        """
        x: (batch, seq_len, input_dim) 或 (batch, input_dim)
        mode: "deq"=隐式不动点求解, "ssm"=显式序列步进
        """
        if x.dim() == 2:
            # 单步DEQ求解
            if mode == "deq":
                h_star, converged, iters, meta = self.deq_solver.solve(x)
                y = self.ssm.C(h_star) + self.ssm.D(x)
                return y, {"mode": "deq", **meta}
            else:
                h = torch.zeros(x.size(0), self.state_dim, device=x.device, dtype=x.dtype)
                h, y = self.ssm.forward_step(h, x)
                return y, {"mode": "ssm", "iterations": 1}
        else:
            # 序列模式
            if mode == "deq":
                # 每步都用DEQ（工程上罕见，这里用于验证）
                outputs = []
                metas = []
                h = torch.zeros(x.size(0), self.state_dim, device=x.device, dtype=x.dtype)
                for t in range(x.size(1)):
                    h, converged, iters, meta = self.deq_solver.solve(x[:, t, :], h_init=h)
                    y = self.ssm.C(h) + self.ssm.D(x[:, t, :])
                    outputs.append(y)
                    metas.append(meta)
                return torch.stack(outputs, dim=1), {"mode": "seq_deq", "steps": len(metas)}
            else:
                y, h = self.ssm.forward_sequence(x)
                return y, {"mode": "seq_ssm"}
