"""
通用工具模块：配置加载、随机种子、度量计算
严格无硬编码，全部从config.yaml读取
"""
import yaml
import torch
import numpy as np
import random


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_attention_entropy(attn_weights):
    """
    计算注意力结构熵：S_att = - sum_ij A_ij log A_ij
    返回每个位置的熵 (batch, seq_len)
    attn_weights: (batch, seq_len, seq_len)
    """
    eps = 1e-9
    attn_weights = attn_weights.clamp(min=eps)
    entropy = -(attn_weights * attn_weights.log()).sum(dim=-1)
    return entropy


def normalize_order_param(entropy, seq_len):
    """
    微观序参量：phi_att = 1 - S_att / log(L)
    phi=0 均匀对称态，phi->1 高度结构化
    """
    logL = np.log(seq_len)
    phi = 1.0 - entropy / logL
    return phi.clamp(0.0, 1.0)


def spectral_radius(matrix):
    """计算矩阵谱半径 rho(A)"""
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)
    eigs = torch.linalg.eigvals(matrix)
    return eigs.abs().max().item()


def compute_wasserstein_approx(mu, nu):
    """
    简化Wasserstein距离近似（Sinkhorn单步）
    mu, nu: (batch, d)
    """
    return (mu - nu).abs().sum(dim=-1).mean()
