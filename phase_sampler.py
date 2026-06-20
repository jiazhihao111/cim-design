"""
连续-离散相变采样器
解决 F-05 连续-离散相变断层
核心机制：
  1. 玻尔兹曼能量场映射: 隐状态 -> 能量 E(h)
  2. SDE破缺噪声注入: dX = -grad E dt + sigma dW
  3. Gumbel-Softmax: 对称性破缺采样，连续能量场 -> 离散Token
对应设计稿："流形到Token的自然对称破缺坍缩"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EnergyMapper(nn.Module):
    """将隐状态映射为玻尔兹曼能量场"""
    def __init__(self, input_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(input_dim, vocab_size)

    def forward(self, h):
        """
        h: (batch, ..., input_dim)
        返回: 能量 E (越高概率越低), 逻辑值 logits
        """
        logits = self.proj(h)
        # 能量 = -logits，softmax概率正比于 exp(-E) = exp(logits)
        energy = -logits
        return energy, logits


class SDENoiseInjector:
    """SDE破缺噪声注入: 提供跨越势垒的第一推动力"""
    def __init__(self, noise_scale, dt=0.01):
        self.noise_scale = noise_scale
        self.dt = dt

    def inject(self, x):
        """欧拉-丸山离散化: x' = x + sigma * sqrt(dt) * N(0,1)"""
        noise = torch.randn_like(x) * self.noise_scale * np.sqrt(self.dt)
        return x + noise


class PhaseSampler(nn.Module):
    """
    连续-离散相变采样器
    实现流形稳态 -> 离散Token的自然结晶
    """
    def __init__(self, input_dim, vocab_size,
                 temperature, sde_noise_scale, gumbel_tau,
                 use_sde=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.gumbel_tau = gumbel_tau
        self.use_sde = use_sde

        self.energy_mapper = EnergyMapper(input_dim, vocab_size)
        self.sde = SDENoiseInjector(sde_noise_scale)

    def forward(self, h, hard=False):
        """
        h: (batch, seq_len, input_dim) 或 (batch, input_dim)
        返回: token_probs, sampled_tokens
        """
        energy, logits = self.energy_mapper(h)

        # SDE破缺噪声注入（对称性破缺）
        if self.use_sde:
            logits = self.sde.inject(logits)

        # 温度缩放（玻尔兹曼分布）
        logits = logits / self.temperature

        # Gumbel-Softmax: 可微分的离散采样
        if self.training or not hard:
            probs = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=hard, dim=-1)
        else:
            # 推理时硬采样
            probs = F.softmax(logits, dim=-1)

        tokens = torch.argmax(probs, dim=-1)
        return probs, tokens, energy

    def sample_crystallize(self, h, num_samples=1):
        """
        模拟"结晶"过程：从连续流形多次采样，选择能量最低（最稳定）的Token
        对应"系统坍缩至高阶全局不动点"
        """
        best_tokens = None
        best_energy = float('inf')
        for _ in range(num_samples):
            _, tokens, energy = self.forward(h, hard=True)
            # 以token位置能量的最小值作为选择标准（简化）
            avg_energy = energy.gather(-1, tokens.unsqueeze(-1)).mean().item()
            if avg_energy < best_energy:
                best_energy = avg_energy
                best_tokens = tokens
        return best_tokens, best_energy
