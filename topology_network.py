"""
拓扑网络模块：小世界网络 + 吸引子盆地
修复：添加拓扑影响矩阵接口，可被L1/L2路由使用
"""
import numpy as np
import torch
import torch.nn as nn


class SmallWorldTopology(nn.Module):
    """
    小世界拓扑网络：局部密集 + 长程稀疏
    修复：添加 compute_influence() 方法，输出拓扑影响矩阵
    """
    def __init__(self, num_nodes, k_nearest, rewire_prob, device="cpu"):
        super().__init__()
        self.num_nodes = num_nodes
        self.k_nearest = k_nearest
        self.rewire_prob = rewire_prob
        self.device = device

        # 构建环形最近邻初始拓扑
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i in range(num_nodes):
            for j in range(1, k_nearest // 2 + 1):
                adj[i, (i + j) % num_nodes] = 1.0
                adj[i, (i - j) % num_nodes] = 1.0

        # Watts-Strogatz 重连
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adj[i, j] == 1.0 and np.random.rand() < rewire_prob:
                    adj[i, j] = 0.0
                    new_target = np.random.randint(0, num_nodes)
                    while new_target == i or adj[i, new_target] == 1.0:
                        new_target = np.random.randint(0, num_nodes)
                    adj[i, new_target] = 1.0

        adj = np.maximum(adj, adj.T)
        self.register_buffer("adj_matrix", torch.from_numpy(adj).to(device))

        # 可学习的边权重
        self.edge_weights = nn.Parameter(torch.randn(num_nodes, num_nodes, device=device) * 0.01)
        self.node_bias = nn.Parameter(torch.zeros(num_nodes, device=device))

        # 输出投影：将拓扑节点映射到模型维度
        self.topo_proj = nn.Linear(num_nodes, 1)

    def effective_weights(self):
        return self.adj_matrix * self.edge_weights

    def forward(self, x):
        """
        x: (batch, num_nodes)
        返回: (batch, num_nodes)
        """
        W = self.effective_weights()
        out = torch.matmul(x, W.t()) + self.node_bias
        return torch.tanh(out)

    def compute_influence(self, seq_len, batch_size=1, device=None):
        """
        计算拓扑影响矩阵 (seq_len, seq_len)
        将64个拓扑节点映射到序列长度的连接矩阵
        
        返回: (batch, seq_len, seq_len) 拓扑连接强度
        """
        if device is None:
            device = self.adj_matrix.device
        
        W = self.effective_weights()  # (64, 64)
        
        # 使用线性插值将64x64拓扑映射到seq_len x seq_len
        influence = torch.zeros(seq_len, seq_len, device=device)
        
        # 映射每个序列位置到最近的拓扑节点
        for i in range(seq_len):
            src_node = int(i * self.num_nodes / seq_len) % self.num_nodes
            for j in range(seq_len):
                dst_node = int(j * self.num_nodes / seq_len) % self.num_nodes
                influence[i, j] = W[src_node, dst_node]
        
        # 归一化
        influence = torch.tanh(influence)
        influence = influence.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, seq, seq)
        
        return influence

    def topology_metrics(self):
        """计算小世界拓扑指标"""
        adj = self.adj_matrix.cpu().numpy()
        n = self.num_nodes
        degrees = adj.sum(axis=1)
        avg_degree = degrees.mean()
        
        clustering = []
        for i in range(n):
            neighbors = np.where(adj[i] > 0)[0]
            if len(neighbors) < 2:
                clustering.append(0.0)
                continue
            links = 0
            for j in neighbors:
                for k in neighbors:
                    if j < k and adj[j, k] > 0:
                        links += 1
            possible = len(neighbors) * (len(neighbors) - 1) / 2
            clustering.append(links / possible if possible > 0 else 0.0)
        
        return {
            "avg_degree": float(avg_degree),
            "avg_clustering": float(np.mean(clustering)),
            "num_edges": int(adj.sum() / 2)
        }


class AttractorBasin(nn.Module):
    """
    吸引子盆地：记忆即流形上的"洼地"
    """
    def __init__(self, topology: SmallWorldTopology):
        super().__init__()
        self.topology = topology

    def energy(self, x):
        W = self.topology.effective_weights()
        b = self.topology.node_bias
        quadratic = -0.5 * (x @ W * x).sum(dim=-1)
        linear = -(b * x).sum(dim=-1)
        return quadratic + linear

    def relax(self, x_init, steps=10, dt=0.1):
        x = x_init.clone()
        trajectory = [x.clone().detach()]
        for _ in range(steps):
            W = self.topology.effective_weights()
            b = self.topology.node_bias
            grad = -(x @ W + b)
            x = x + dt * grad
            x = torch.tanh(x)
            trajectory.append(x.clone().detach())
        return x, trajectory
