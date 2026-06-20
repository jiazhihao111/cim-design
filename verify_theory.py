"""
理论验证实验集
基于设计稿六层递进验证体系，实现可复现的数值验证
"""
import os
import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import load_config, set_seed, compute_attention_entropy, normalize_order_param, spectral_radius
from topology_network import SmallWorldTopology, AttractorBasin
from l1_micro_fusion import SRAMFusionEngine
from l2_plastic_router import PlasticityMoELayer
from l3_relaxation import RelaxationEngine, SSMCell
from phase_sampler import PhaseSampler


# ============================================================
# 实验1: 涌现性验证 —— 三元核心+时序错位=高层涌现
# ============================================================
def verify_emergence(cfg, save_dir="./results"):
    """
    验证核心命题：同质极简规则 + 统一约束 + 差异化时序 -> 涌现
    方法：构建简单元胞自动机式网络，观察时序错位是否导致有序结构涌现
    """
    print("\n[实验1] 涌现性验证：三元核心定理")
    os.makedirs(save_dir, exist_ok=True)
    set_seed(cfg["system"]["seed"])

    num_nodes = cfg["topology"]["num_nodes"]
    steps = 60
    trials = cfg["verify"]["emergence_trials"]

    # 三种条件对比
    results = {"full": [], "no_breaking": [], "no_sequence": []}

    for condition in ["full", "no_breaking", "no_sequence"]:
        order_params = []
        for t in range(trials):
            topo = SmallWorldTopology(
                num_nodes=num_nodes,
                k_nearest=cfg["topology"]["k_nearest"],
                rewire_prob=cfg["topology"]["rewire_prob"] if condition != "no_breaking" else 0.0,
                device="cpu"
            )
            basin = AttractorBasin(topo)

            # 初始高熵态
            x = torch.randn(1, num_nodes) * 0.5

            # 差异化时序：不同相位启动
            if condition != "no_sequence":
                phase_mask = torch.rand(1, num_nodes) > 0.3  # 30%节点延迟启动
            else:
                phase_mask = torch.ones(1, num_nodes)  # 全部同步

            diffs = []
            for step in range(steps):
                if condition != "no_sequence":
                    x_new, _ = basin.relax(x, steps=1, dt=0.2)
                    x = torch.where(phase_mask, x_new, x)
                else:
                    x, _ = basin.relax(x, steps=1, dt=0.2)

                # 序参量：状态分化度（标准差）+ 能量下降的组合指标
                # 分化度高且能量低 = 涌现强烈
                std = x.std(dim=-1).item()
                E = basin.energy(x).item()
                order = std - E  # 高分化 + 低能量 = 高序参量
                diffs.append(order)

            order_params.append(diffs)

        arr = np.array(order_params)
        results[condition] = arr.mean(axis=0)

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(results["full"], label="full (breaking+rule+seq)", color="red", linewidth=2)
    plt.plot(results["no_breaking"], label="no_breaking (rewire=0)", color="blue", linestyle="--")
    plt.plot(results["no_sequence"], label="no_sequence (sync)", color="green", linestyle="-.")
    plt.xlabel("Iteration")
    plt.ylabel("Order Parameter (std - energy)")
    plt.title("Exp1: Emergence via Triadic Core")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "exp1_emergence.png"), dpi=150)
    plt.close()

    # 判定：完整条件下序参量应显著高于对照组
    full_final = results["full"][-10:].mean()
    no_break_final = results["no_breaking"][-10:].mean()
    no_seq_final = results["no_sequence"][-10:].mean()

    verdict = full_final > max(no_break_final, no_seq_final) * 1.03 and full_final > no_seq_final * 1.1
    print(f"  Full triadic final order: {full_final:.4f}")
    print(f"  No breaking final order: {no_break_final:.4f}")
    print(f"  No sequence final order: {no_seq_final:.4f}")
    print(f"  Result: {'PASS' if verdict else 'FAIL'} (full > no-break by 3% & full > no-seq by 10%)")
    return verdict, results


# ============================================================
# 实验2: 收敛性验证 —— DEQ不动点 + 谱半径约束 + 断路器
# ============================================================
def verify_convergence(cfg, save_dir="./results"):
    """
    验证核心命题：DEQ隐式求解收敛，谱归一化防止发散，断路器兜底
    """
    print("\n[实验2] 收敛性验证：DEQ + 谱归一化 + 断路器")
    os.makedirs(save_dir, exist_ok=True)
    set_seed(cfg["system"]["seed"])

    l3c = cfg["l3_relaxation"]
    trials = cfg["verify"]["convergence_trials"]

    # 对比：启用 vs 禁用谱归一化+断路器
    modes = [(True, True, "完整保护"), (False, False, "无保护")]
    stats = {}

    for use_sn, use_cb, name in modes:
        converged_count = 0
        fallback_count = 0
        iter_list = []
        residual_list = []

        for _ in range(trials):
            engine = RelaxationEngine(
                input_dim=cfg["l1_fusion"]["d_model"],
                state_dim=l3c["ssm_state_dim"],
                dt=l3c["dt"],
                deq_max_iter=l3c["deq_max_iter"],
                deq_tol=l3c["deq_tol"],
                spectral_radius_threshold=l3c["spectral_radius_threshold"],
                spectral_norm_gamma=l3c["spectral_norm_gamma"],
                use_spectral_norm=use_sn,
                use_circuit_breaker=use_cb,
                circuit_breaker_max_iter=l3c["circuit_breaker_max_iter"]
            )
            # 临界不稳定初始化：A的谱半径接近0.95~1.05之间
            with torch.no_grad():
                # 先随机初始化，然后缩放使谱半径接近临界值
                target_rho = 0.98 + np.random.rand() * 0.1  # 0.98~1.08
                current_rho = engine.ssm.spectral_radius()
                if current_rho > 0:
                    engine.ssm.A.data *= (target_rho / current_rho)

            x = torch.randn(1, cfg["l1_fusion"]["d_model"])
            h, converged, iters, meta = engine.deq_solver.solve(x)

            if meta["converged"]:
                converged_count += 1
            if meta.get("fallback_triggered", False):
                fallback_count += 1
            iter_list.append(iters)
            residual_list.append(meta["final_residual"])

        stats[name] = {
            "converged_rate": converged_count / trials,
            "fallback_rate": fallback_count / trials,
            "avg_iters": np.mean(iter_list),
            "max_iters": max(iter_list),
            "avg_residual": np.mean(residual_list)
        }

    # 输出
    for name, s in stats.items():
        print(f"  [{name}] converge={s['converged_rate']:.2%}, fallback={s['fallback_rate']:.2%}, "
              f"avg_iter={s['avg_iters']:.1f}, avg_resid={s['avg_residual']:.2e}")

    # 判定：保护组有更高收敛率，或能安全截断（无无限死循环）
    protected = stats["完整保护"]
    unprotected = stats["无保护"]
    verdict = (protected["converged_rate"] >= unprotected["converged_rate"] and
               protected["max_iters"] <= l3c["circuit_breaker_max_iter"] + 1)
    print(f"  Result: {'PASS' if verdict else 'FAIL'} (protected converges no worse & no deadlock)")
    return verdict, stats


# ============================================================
# 实验3: 存算效率验证 —— 访存复杂度对比
# ============================================================
def verify_io_efficiency(cfg, save_dir="./results"):
    """
    验证核心命题：L1分块融合后，HBM访存从O(N^2)降至近似O(N)
    """
    print("\n[实验3] 存算效率验证：访存复杂度对比")
    os.makedirs(save_dir, exist_ok=True)

    l1c = cfg["l1_fusion"]
    seq_lens = cfg["verify"]["io_compare_seq_lens"]
    d = l1c["d_model"]

    standard_ios = []
    fused_ios = []

    engine = SRAMFusionEngine(
        sram_capacity=l1c["sram_capacity"],
        block_size=l1c["block_size"],
        d_model=d,
        entropy_threshold_high=l1c["entropy_threshold_high"],
        entropy_threshold_low=l1c["entropy_threshold_low"],
        async_prefetch_window=l1c["async_prefetch_window"],
        flash_attention_sim=True
    )

    for N in seq_lens:
        std_io = engine.standard_attention_io(N)
        fused_io = engine.fused_attention_io(N)
        standard_ios.append(std_io)
        fused_ios.append(fused_io)

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(seq_lens, standard_ios, 'o-', label="标准Attention O(N^2)", color="blue")
    plt.plot(seq_lens, fused_ios, 's-', label="融合引擎 O(N)近似", color="red")
    plt.xlabel("序列长度 N")
    plt.ylabel("HBM访存量 (元素数)")
    plt.title("实验3：存算融合访存复杂度对比")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "exp3_io_efficiency.png"), dpi=150)
    plt.close()

    # 验证：长序列下降低倍数
    ratios = [s / f for s, f in zip(standard_ios, fused_ios)]
    print(f"  序列长度: {seq_lens}")
    print(f"  标准访存: {standard_ios}")
    print(f"  融合访存: {fused_ios}")
    print(f"  降低倍数: {[f'{r:.2f}x' for r in ratios]}")

    verdict = ratios[-1] >= 2.0  # 最长序列至少降低2倍
    print(f"  验证结果: {'通过' if verdict else '未通过'} (长序列降低倍数≥2)")
    return verdict, {"seq_lens": seq_lens, "ratios": ratios}


# ============================================================
# 实验4: 可塑性稳定性验证 —— EWC防止灾难性遗忘
# ============================================================
def verify_plasticity_stability(cfg, save_dir="./results"):
    """
    验证核心命题：启用EWC+EMA后，在线TTT不会导致权重漂移失控
    """
    print("\n[实验4] 可塑性稳定性验证：EWC + EMA 抗遗忘")
    os.makedirs(save_dir, exist_ok=True)
    set_seed(cfg["system"]["seed"])

    l2c = cfg["l2_router"]
    d = cfg["l1_fusion"]["d_model"]
    steps = cfg["verify"]["plasticity_steps"]

    conditions = [
        (True, True, "EWC+EMA"),
        (False, True, "仅EMA"),
        (True, False, "仅EWC"),
        (False, False, "无保护")
    ]
    stats = {}

    for use_ewc, use_ema, name in conditions:
        layer = PlasticityMoELayer(
            num_experts=l2c["num_experts"],
            top_k=l2c["top_k"],
            d_model=d,
            plasticity_lr=l2c["plasticity_lr"],
            weight_clip=l2c["weight_clip"],
            use_ewc=use_ewc,
            ewc_lambda=l2c["ewc_lambda"],
            use_ema=use_ema,
            ema_momentum=l2c["ema_momentum"]
        )
        layer.train()

        # 预训练阶段：为EWC计算Fisher信息矩阵（模拟预训练后保护核心知识）
        if use_ewc and layer.ewc is not None:
            pretrain_data = [torch.randn(2, 8, d) for _ in range(20)]
            layer.ewc.compute_fisher(layer, pretrain_data, num_samples=20)

        # 记录初始权重
        init_weights = {n: p.clone().detach() for n, p in layer.named_parameters()}

        # 模拟在线TTT多步（分布外输入）
        deltas = []
        for _ in range(steps):
            x = torch.randn(2, 8, d) * 2.0  # 增大方差模拟OOD输入
            _, _ = layer(x, enable_ttt=True)

            # 计算与初始权重的总偏差
            total_delta = sum((p - init_weights[n]).norm().item()
                              for n, p in layer.named_parameters() if n in init_weights)
            deltas.append(total_delta)

        stats[name] = {
            "final_delta": deltas[-1],
            "delta_trend": deltas
        }
        print(f"  [{name}] final weight drift: {deltas[-1]:.4f}")

    # 绘图
    plt.figure(figsize=(8, 5))
    for name, s in stats.items():
        plt.plot(s["delta_trend"], label=name)
    plt.xlabel("TTT更新步")
    plt.ylabel("累计权重漂移 L2范数")
    plt.title("实验4：可塑性稳定性对比")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "exp4_plasticity.png"), dpi=150)
    plt.close()

    # EWC+EMA应低于无保护
    protected = stats["EWC+EMA"]["final_delta"]
    unprotected = stats["无保护"]["final_delta"]
    verdict = protected < unprotected * 0.8
    print(f"  Result: {'PASS' if verdict else 'FAIL'} (protected drift < 80% of unprotected)")
    return verdict, stats


# ============================================================
# 实验5: 连续-离散相变验证 —— SDE噪声驱动对称性破缺
# ============================================================
def verify_phase_transition(cfg, save_dir="./results"):
    """
    验证核心命题：SDE注入的破缺噪声导致Token分布从均匀对称态坍缩为结构化态
    """
    print("\n[实验5] 相变验证：SDE对称性破缺采样")
    os.makedirs(save_dir, exist_ok=True)
    set_seed(cfg["system"]["seed"])

    psc = cfg["phase_sampler"]
    d = cfg["l1_fusion"]["d_model"]
    sampler_with = PhaseSampler(d, psc["vocab_size"], psc["temperature"],
                                psc["sde_noise_scale"], psc["gumbel_tau"], use_sde=True)
    sampler_without = PhaseSampler(d, psc["vocab_size"], psc["temperature"],
                                   0.0, psc["gumbel_tau"], use_sde=False)

    h = torch.randn(1, 16, d)  # 连续隐状态

    # 多次采样统计熵
    ents_with = []
    ents_without = []
    for _ in range(50):
        probs_w, _, _ = sampler_with(h)
        ents_with.append(-(probs_w * (probs_w + 1e-9).log()).sum(dim=-1).mean().item())

        probs_wo, _, _ = sampler_without(h)
        ents_without.append(-(probs_wo * (probs_wo + 1e-9).log()).sum(dim=-1).mean().item())

    # 计算与均匀分布的KL散度：偏离越远，破缺越明显
    uniform = torch.ones_like(probs_w) / psc["vocab_size"]
    kl_with = F.kl_div(uniform.log(), probs_w, reduction="batchmean").item()
    kl_without = F.kl_div(uniform.log(), probs_wo, reduction="batchmean").item()

    # 绘图：典型概率分布
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(psc["vocab_size"]), probs_w[0, 0].detach().numpy(), color="red")
    plt.title("With SDE noise")
    plt.xlabel("Token ID")
    plt.ylabel("Probability")

    plt.subplot(1, 2, 2)
    plt.bar(range(psc["vocab_size"]), probs_wo[0, 0].detach().numpy(), color="blue")
    plt.title("Without SDE noise")
    plt.xlabel("Token ID")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "exp5_phase_transition.png"), dpi=150)
    plt.close()

    print(f"  KL(uniform||with SDE): {kl_with:.4f}")
    print(f"  KL(uniform||without SDE): {kl_without:.4f}")
    verdict = kl_with > kl_without * 1.01  # SDE使分布更远离均匀（破缺）
    print(f"  Result: {'PASS' if verdict else 'FAIL'} (SDE drives distribution away from uniform)")
    return verdict, {"kl_with": kl_with, "kl_without": kl_without}


# ============================================================
# 主验证入口
# ============================================================
def run_all_verifications():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = load_config(cfg_path)
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("纯软件模拟存算一体拓扑流形架构 —— 理论验证实验集")
    print("=" * 70)

    results = {}
    results["emergence"] = verify_emergence(cfg, save_dir)
    results["convergence"] = verify_convergence(cfg, save_dir)
    results["io_efficiency"] = verify_io_efficiency(cfg, save_dir)
    results["plasticity"] = verify_plasticity_stability(cfg, save_dir)
    results["phase_transition"] = verify_phase_transition(cfg, save_dir)

    print("\n" + "=" * 70)
    print("验证总览")
    print("=" * 70)
    all_pass = True
    for name, (verdict, _) in results.items():
        status = "通过" if verdict else "未通过"
        print(f"  {name:20s}: {status}")
        if not verdict:
            all_pass = False
    print("=" * 70)
    print(f"总体结论: {'全部验证通过，架构理论自洽' if all_pass else '部分验证未通过，需进一步调优'}")
    print(f"结果图表保存至: {save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_verifications()
