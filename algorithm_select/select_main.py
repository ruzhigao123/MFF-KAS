
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.loader import DataLoader
import sys
import numpy as np
from tqdm import tqdm
import json

# --- 环境配置 ---
BASE_DIR = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master"
sys.path.append(BASE_DIR)
from select_model import AlgorithmSelector
from select_dataset import AlgorithmSelectDataset, load_labels_from_json
from algorithm_utils import extract_ast_meta_features
try:
    from src.config import FLAGS as SRC_FLAGS
    CONFIG_TARGET_M_ID = getattr(SRC_FLAGS, "target_m_id", None)
    CONFIG_TRAIN_SKIP_ALL_ZERO = bool(getattr(SRC_FLAGS, "train_skip_all_zero_config", True))
    CONFIG_TEST_SKIP_ALL_ZERO = bool(getattr(SRC_FLAGS, "test_skip_all_zero_config", True))
except Exception:
    CONFIG_TARGET_M_ID = None
    CONFIG_TRAIN_SKIP_ALL_ZERO = True
    CONFIG_TEST_SKIP_ALL_ZERO = True

TARGET_NAMES = ['perf', 'lut', 'ff', 'dsp', 'bram']
MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'doitgen',
               'mvt', 'fdtd-2d', 'gemver', 'gemm-p', 'gesummv',
               'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d']


from sklearn.metrics import mean_squared_error
from collections import defaultdict
def calculate_pipeline_rmse(selector, loader, device, target_names, k=3):
    """
    计算 Top-3 择优下的物理误差 (MSE/RMSE)
    按 Kernel 划分统计，并输出总平均值
    """
    selector.eval()

    # 存储结构: {kernel_name: {metric_name: {'true': [], 'pred_best_in_k': []}}}
    stats = defaultdict(lambda: defaultdict(lambda: {'true': [], 'pred_best_in_k': []}))

    # 指标名称
    metric_names = target_names  # ['Perf', 'LUT', 'FF', 'DSP', 'BRAM']

    with torch.no_grad():
        for data, meta, m_id, error_vectors in loader:
            data = data.to(device)
            meta = meta.to(device)
            m_id = m_id.to(device)
            # error_vectors 实际上包含了所有工具在该样本上的真实物理值 (Ground Truth)
            error_vectors = error_vectors.to(device)

            # 1. 推荐器预测 Top-K
            logits, _ = selector(data, meta, m_id)
            _, topk_indices = torch.topk(logits, k, dim=1)  # [Batch, K]

            # 2. 从 Top-K 候选专家中寻找真实物理值最优的 (模拟 Top-3 择优后的结果)
            # 注意：在真实推理中我们会用专家模型预测，但在这里我们直接用 error_vectors 里的真值
            # 这样算出来的是“推荐器所能达到的理论最优物理误差”
            topk_true_values = torch.gather(error_vectors, 1, topk_indices)  # [Batch, K]
            best_val_in_k, _ = topk_true_values.min(dim=1)  # 假设越小越优

            # 3. 这里的 true_best 则是全局 7 个专家里的绝对最优值
            true_best_val, _ = error_vectors.min(dim=1)

            # 4. 转换数据并归档到对应的 Kernel
            # 假设你的 data 对象里存了 kernel 名称，如果没有，请确保在 dataset 加载时放入 data.kernel
            # 如果没有 data.kernel，我们可以从 data.file 或者其他属性获取
            for i in range(data.num_graphs):
                # 尝试获取 kernel 名，默认 fallback 为 'Unknown'
                k_name = getattr(data[i], 'kernel', 'Unknown')
                m_idx = m_id[i].item()
                m_name = metric_names[m_idx]

                # 我们关心的是：选中的那个值(best_val_in_k) 与 绝对最优值(true_best_val) 的物理差距
                # 或者：选中的值(best_val_in_k) 与 你想要对比的基准值 的差距
                # 这里按照 RMSE 定义：pred (选中的值) vs true (绝对最优值)
                stats[k_name][m_name]['true'].append(true_best_val[i].item())
                stats[k_name][m_name]['pred_best_in_k'].append(best_val_in_k[i].item())

    # --- 打印统计结果 ---
    print("\n" + "=" * 50)
    print(f"详细物理误差报告 (Top-{k} Selection Mode)")
    print("=" * 50)

    all_total_true = []
    all_total_pred = []

    for k_name, metrics in stats.items():
        print(f"\nKernel: {k_name}")
        print("-" * 30)
        for m_name, vals in metrics.items():
            t = np.array(vals['true'])
            p = np.array(vals['pred_best_in_k'])

            mse = mean_squared_error(t, p)
            rmse = np.sqrt(mse)

            print(f"  [{m_name:4s}] MSE: {mse:10.4f} | RMSE: {rmse:10.4f}")

            all_total_true.extend(vals['true'])
            all_total_pred.extend(vals['pred_best_in_k'])

    # 计算总平均
    overall_mse = mean_squared_error(all_total_true, all_total_pred)
    overall_rmse = np.sqrt(overall_mse)

    print("\n" + "=" * 50)
    print(f"OVERALL AVERAGE -> MSE: {overall_mse:.4f} | RMSE: {overall_rmse:.4f}")
    print("=" * 50 + "\n")

    return overall_rmse


def get_train_val_test_sets(all_aligned_info):
    """
    与 src/train.py 对齐：
    - all_kernels = MACHSUITE + poly
    - np.random.seed(64)
    - 随机打乱后前15个作为训练 kernels，后7个作为 inference/val kernels
    """
    all_kernels = MACHSUITE_KERNEL + poly_KERNEL
    num_train_kernels = 15
    np.random.seed(64)
    shuffled_indices = np.random.permutation(len(all_kernels))
    train_kernel_names = set([all_kernels[i] for i in shuffled_indices[:num_train_kernels]])
    inference_kernel_names = set([all_kernels[i] for i in shuffled_indices[num_train_kernels:]])

    train_list, val_list = [], []
    if all_aligned_info:
        print(f"DEBUG: key[0]={all_aligned_info[0]['key'][0]}, key[1]={all_aligned_info[0]['key'][1]}")
        print(f">>> Train kernels (15): {sorted(list(train_kernel_names))}")
        print(f">>> Inference/Val kernels (7): {sorted(list(inference_kernel_names))}")

    for info in all_aligned_info:
        k_name = info['key'][1]
        if k_name in train_kernel_names:
            train_list.append(info)
        elif k_name in inference_kernel_names:
            val_list.append(info)

    print(f">>> 划分完成 (aligned with src/train.py): Train={len(train_list)}, Val={len(val_list)}")
    return train_list, val_list


def extract_hybrid_meta(ast_path):
    JSON_REPO = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_datasets/code/ast_output"
    PT_REPO = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_datasets/code/ast_final_dataset_1"
    path_parts = str(ast_path).replace('\\', '/').split('/')
    try:
        file_prefix = path_parts[-1].split('.')[0]
        kernel_name = path_parts[-2]
        dataset_name = path_parts[-3]
    except:
        file_prefix = os.path.basename(ast_path).split('.')[0]
        kernel_name, dataset_name = "", ""
    base_9 = torch.zeros(9)
    target_json = os.path.join(JSON_REPO, dataset_name, kernel_name, f"{file_prefix}.ast.json")
    if os.path.exists(target_json):
        with open(target_json, 'r') as f:
            base_9 = extract_ast_meta_features(json.load(f))
    tail_16 = torch.zeros(16)
    target_pt = os.path.join(PT_REPO, dataset_name, kernel_name, f"{file_prefix}.ast.pt")
    if os.path.exists(target_pt):
        d_obj = torch.load(target_pt, map_location='cpu')
        feat = d_obj.x if hasattr(d_obj, 'x') else d_obj
        feat_t = torch.as_tensor(feat).float()
        # 优先使用节点级 16 维硬件特征做“pragma 聚合”，更直接反映配置 pragma 信息
        if feat_t.dim() == 2 and feat_t.shape[1] >= 16:
            hw16 = feat_t[:, -16:]  # [num_nodes, 16]
            pragma_mask = hw16[:, 0] > 0.5  # is_pragma_or_accel
            if pragma_mask.any():
                pragma_hw = hw16[pragma_mask]
                tail_16 = pragma_hw.mean(dim=0)
                # 对影响性能最强的 pragma 强度特征，用 max 保留峰值信号
                for idx in [1, 2, 3, 5, 6]:
                    tail_16[idx] = pragma_hw[:, idx].max()
                # 节点类型占比（仅在 pragma 节点集合内）
                tail_16[12] = (pragma_hw[:, 12] > 0).float().mean()  # is_loop_node ratio
                tail_16[13] = (pragma_hw[:, 13] > 0).float().mean()  # is_branch_node ratio
            else:
                tail_16 = hw16.mean(dim=0)
        else:
            full_feat = feat_t.flatten()
            tail_16 = full_feat[-16:] if full_feat.shape[0] >= 16 else full_feat[:16]
    return torch.cat([base_9, tail_16], dim=0)


# --- 核心：修正后的训练逻辑，增加 Ratio (性能比) 统计 ---
import torch
import torch.nn.functional as F
from torch.distributions import Categorical



import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.distributions import Categorical
import copy


def train_ppo_style(
        model, loader, optimizer, device, entropy_weight, epoch, max_epochs=1000,
        phase2_active=False, teacher_model=None,
        keep_top3_w=0.20, sort_in_top3_w=0.25):  # 引入 soft label + margin weighting
    model.train()
    total_loss, total_reward, total_ratio = 0, 0, 0
    top1_hits, total_samples = 0, 0

    # --- 动态超参数 ---
    if epoch < 100:
        temperature = 2.0
        rank_loss_weight = 0.8
    else:
        temperature = 1.0
        rank_loss_weight = 0.5

    for data, h_s, h_a, meta, m_id, error_vectors in loader:
        data = data.to(device)
        # h_s, h_a, m_id, error_vectors = [x.to(device) for x in [h_s, h_a, m_id, error_vectors]]   # 未加入25维度元特征
        h_s, h_a, meta, m_id, error_vectors = [x.to(device) for x in [h_s, h_a, meta, m_id, error_vectors]]
        optimizer.zero_grad()

        # 1. 前向传播
        # logits = model(data, h_s, h_a, m_id)
        logits = model(data, h_s, h_a, meta, m_id)

        # 2. 强化学习采样
        probs = F.softmax(logits / temperature, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)

        with torch.no_grad():
            # 获取真实最优值和全排名
            true_best_val, true_best_idx = error_vectors.min(dim=1)
            # 获取真实排名索引 (误差从小到大排序)
            _, true_sorted_idx = torch.sort(error_vectors, dim=1, descending=False)
            sorted_errors, _ = torch.sort(error_vectors, dim=1, descending=False)
            best_err = sorted_errors[:, 0]
            second_err = sorted_errors[:, 1]

            # 计算采样动作在真实排名中的位置 (0 是第一名)
            # 找到 action 在 true_sorted_idx 每一行的位置
            action_rank = (true_sorted_idx == action.unsqueeze(1)).nonzero()[:, 1]

            # 使用相对 margin 判断样本是否“近并列最优”
            # margin 越小，表示 top1/top2 越难分，训练时应降低其硬惩罚强度
            rel_margin = (second_err - best_err) / (best_err.abs() + 1e-6)
            sample_weight = torch.ones_like(rel_margin)
            sample_weight = torch.where(rel_margin < 0.01, torch.full_like(sample_weight, 0.3), sample_weight)
            sample_weight = torch.where((rel_margin >= 0.01) & (rel_margin < 0.03),
                                        torch.full_like(sample_weight, 0.5), sample_weight)
            sample_weight = torch.where(rel_margin >= 0.03, torch.full_like(sample_weight, 1.0), sample_weight)

        # 3. 计算阶梯奖励 (Top-K Reward Guidance)
        selected_loss = error_vectors.gather(1, action.unsqueeze(1)).squeeze()
        # 基础奖励：Ratio
        base_reward = (true_best_val + 1e-6) / (selected_loss + 1e-6)

        # 附加奖励：根据排名给分 (让前三向真实靠拢)
        # 如果选到第1名给2分，第2名给1分，第3名给0.5分，其余不给分甚至扣分
        rank_bonus = torch.zeros_like(base_reward)
        rank_bonus[action_rank == 0] = 2.0
        rank_bonus[action_rank == 1] = 1.0
        rank_bonus[action_rank == 2] = 0.5
        rank_bonus[action_rank > 2] = -0.2

        combined_reward = base_reward + rank_bonus

        # Perf 先验奖励：若选择了含 AST 的专家，额外加分（增强 Top-K 中 AST 专家出现概率）
        # 含 AST 专家集合: ast(5), all(0), ast_and_cdfg(1), ast_and_text(2)
        perf_mask = (m_id == 0)
        if perf_mask.any():
            ast_pref_actions = (
                (action == 5) | (action == 0) | (action == 1) | (action == 2)
            )
            perf_ast_bonus = (perf_mask & ast_pref_actions).float() * 0.25
            combined_reward = combined_reward + perf_ast_bonus

        # 标准化 Advantage
        if combined_reward.shape[0] > 1:
            adv = (combined_reward - combined_reward.mean()) / (combined_reward.std() + 1e-8)
        else:
            adv = combined_reward - 0.5

        # 4. 计算辅助对齐损失 (Soft Label + Rank Loss)
        # 4.1 soft label：小 margin 样本给更软的目标，降低“第一第二互换”惩罚
        per_sample_temp = torch.where(rel_margin < 0.03, torch.full_like(rel_margin, 0.8),
                                      torch.full_like(rel_margin, 0.25))
        target_dist = F.softmax(-error_vectors / per_sample_temp.unsqueeze(1), dim=1)
        log_probs_all = F.log_softmax(logits, dim=1)
        loss_soft = -(target_dist * log_probs_all).sum(dim=1)
        loss_soft = (loss_soft * sample_weight).sum() / (sample_weight.sum() + 1e-8)

        # 4.2 hard negative 排序损失：明确拉开“真第一”与“最危险对手”的间隔
        true_logit = logits.gather(1, true_best_idx.unsqueeze(1)).squeeze(1)
        masked_logits = logits.clone()
        masked_logits.scatter_(1, true_best_idx.unsqueeze(1), float('-inf'))
        hard_neg_idx = masked_logits.argmax(dim=1)
        hard_neg_logit = logits.gather(1, hard_neg_idx.unsqueeze(1)).squeeze(1)
        adaptive_margin = torch.where(rel_margin < 0.03, torch.full_like(rel_margin, 0.03),
                                      torch.full_like(rel_margin, 0.10))
        loss_rank = F.relu(adaptive_margin - (true_logit - hard_neg_logit))
        loss_rank = (loss_rank * sample_weight).sum() / (sample_weight.sum() + 1e-8)

        # 4.3 Top-1 监督损失：直接优化第一名排序能力
        loss_ce = F.cross_entropy(logits, true_best_idx)

        # 4.4 先验知识正则（训练阶段生效）：
        # 当指标是 perf(m_id==0) 时，鼓励概率质量更多分配给“含 AST”专家：
        # ast(5), all(0), ast_and_cdfg(1), ast_and_text(2)
        perf_mask = (m_id == 0)
        if perf_mask.any():
            perf_probs = probs[perf_mask]  # [N_perf, 7]
            ast_pref_mass = (
                perf_probs[:, 5] +  # ast
                perf_probs[:, 0] +  # all
                perf_probs[:, 1] +  # ast_and_cdfg
                perf_probs[:, 2]    # ast_and_text
            )
            # 最大化 ast_pref_mass <=> 最小化 -log(ast_pref_mass)
            loss_perf_prior = -torch.log(ast_pref_mass.clamp(min=1e-8)).mean()
        else:
            loss_perf_prior = torch.tensor(0.0, device=logits.device)

        # 5. 总损失融合
        loss_rl = -(log_prob * adv.detach()).mean()
        loss_entropy = - entropy_weight * m.entropy().mean()

        # 总 Loss = Top1 监督 + 排序对齐 + soft label + 轻量 RL 探索 + 熵正则 + perf先验
        # 增强偏好：前期更强，后期回落，减少对真实误差监督的过度覆盖
        if epoch < 120:
            perf_prior_w = 0.20
        elif epoch < 300:
            perf_prior_w = 0.14
        else:
            perf_prior_w = 0.10
        loss = (
            0.55 * loss_ce
            + rank_loss_weight * loss_rank
            + 0.35 * loss_soft
            + 0.15 * loss_rl
            + loss_entropy
            + perf_prior_w * loss_perf_prior
        )

        # 6) 二阶段约束（freeze Top-3 set + focus ordering inside that set）
        if phase2_active and (teacher_model is not None):
            with torch.no_grad():
                teacher_logits = teacher_model(data, h_s, h_a, meta, m_id)
                _, teacher_top3_idx = torch.topk(teacher_logits, k=3, dim=1)  # [B,3]

            # 6.1 保持 teacher Top-3 集合：鼓励概率质量集中在 teacher 的 top-3 上
            probs_curr = F.softmax(logits, dim=1)
            top3_mass = probs_curr.gather(1, teacher_top3_idx).sum(dim=1)  # [B]
            loss_keep_top3 = -torch.log(top3_mass.clamp(min=1e-8)).mean()

            # 6.2 在 teacher Top-3 内做排序：选真实误差最小者
            top3_logits_student = logits.gather(1, teacher_top3_idx)  # [B,3]
            top3_true_errors = error_vectors.gather(1, teacher_top3_idx)  # [B,3]
            top3_target_pos = top3_true_errors.argmin(dim=1)  # [B]
            loss_sort_top3 = F.cross_entropy(top3_logits_student, top3_target_pos)

            loss = loss + keep_top3_w * loss_keep_top3 + sort_in_top3_w * loss_sort_top3

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # --- 统计指标（训练主指标改为 Top-1）---
        pred_top1 = logits.argmax(dim=1)
        top1_hits += (pred_top1 == true_best_idx).sum().item()
        total_samples += data.num_graphs

        pred_top1_err = error_vectors.gather(1, pred_top1.unsqueeze(1)).squeeze()
        current_ratio = (true_best_val + 1e-6) / (pred_top1_err + 1e-6)

        total_ratio += current_ratio.mean().item()
        total_loss += loss.item()
        total_reward += combined_reward.mean().item()

    avg_reward = total_reward / len(loader)
    avg_acc = top1_hits / total_samples
    avg_ratio = total_ratio / len(loader)

    return avg_reward, avg_acc, avg_ratio

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import torch
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def evaluate_performance(model, loader, device, k=3, metric_name=None, lock_top3_for_eval=False, frozen_top3_map=None):
    """
    评估函数：适配三特征融合版模型
    1. 统计 Top-1, Top-2, Top-3 独立性能。
    2. 优先专家选择策略 (Priority Selection: 0 > 1,2,3 > Top-1)。
    3. 指标驱动专家策略 (Expert-Driven: Perf->Exp0, Area->Exp2, else Top-1)。
    4. 统计“随机从前K中选一个”的性能。
    5. 返回 Top-K 候选池中的最优潜力性能。
    """
    model.eval()

    # --- 映射表定义区域 ---
    expert_names = ["Expert_0", "Expert_1", "Expert_2", "Expert_3", "Expert_4", "Expert_5", "Expert_6"]
    metric_map = {0: "Perf", 1: "LUT", 2: "FF", 3: "DSP", 4: "BRAM"}
    print_sample_count = 0

    # 统计容器
    hits = {1: 0, 2: 0, 3: 0}
    ratios = {1: 0.0, 2: 0.0, 3: 0.0}

    # 策略 1: 原始优先策略 (0 > 1,2,3 > Top-1)
    total_hits_priority = 0
    total_ratio_priority = 0.0

    # 策略 2: 新增指标驱动策略 (Perf->0, Area->2, else Top-1)
    total_hits_expert_driven = 0
    total_ratio_expert_driven = 0.0

    total_hits_random_k = 0
    total_ratio_random_k = 0.0
    total_hits_best_in_topk = 0
    total_ratio_best_in_topk = 0.0

    # margin 诊断：用于判断 top1/top2 是否大量“近并列”
    margin_buckets = {
        "lt_1": {"count": 0, "top1_hits": 0, "top1_ratio_sum": 0.0},
        "1_to_3": {"count": 0, "top1_hits": 0, "top1_ratio_sum": 0.0},
        "ge_3": {"count": 0, "top1_hits": 0, "top1_ratio_sum": 0.0},
    }

    total_count = 0
    m_ratios = {i: [] for i in range(5)}
    if frozen_top3_map is None:
        frozen_top3_map = {}

    for data, h_s, h_a, meta, m_id, error_vectors in loader:
        data = data.to(device)
        h_s, h_a, meta, m_id, error_vectors = [x.to(device) for x in [h_s, h_a, meta, m_id, error_vectors]]

        # 模型前向传播
        logits = model(data, h_s, h_a, meta, m_id)
        # Logit bias（soft prior）：仅对 perf 样本加小偏置，不做硬替换
        logits_eval = logits.clone()
        perf_rows = (m_id == 0)
        if perf_rows.any():
            b0, b1, b5, b2 = 0.25, 0.18, 0.12, 0.08
            logits_eval[perf_rows, 0] += b0  # all
            logits_eval[perf_rows, 1] += b1  # ast_and_cdfg
            logits_eval[perf_rows, 5] += b5  # ast
            logits_eval[perf_rows, 2] += b2  # ast_and_text
        batch_size = data.num_graphs
        total_count += batch_size

        # 获取真实最优值
        true_best_val, true_best_idx = error_vectors.min(dim=1)

        # 获取预测 Top-K（基于加偏置后的 logits_eval）
        _, topk_indices = torch.topk(logits_eval, k, dim=1)
        top5_k = min(5, logits.size(1))
        _, top5_indices = torch.topk(logits_eval, top5_k, dim=1)
        _, true_topk_indices = torch.sort(error_vectors, dim=1, descending=False)

        # 二阶段评估锁定：对 val/test 样本固定 Top-3 集合（按样本键缓存）
        if lock_top3_for_eval:
            topk_indices_locked = topk_indices.clone()
            for b in range(batch_size):
                if hasattr(data, 'kernel'):
                    k_name = str(data.kernel[b])
                else:
                    k_name = str(getattr(data[b], 'kernel', 'Unknown'))
                if hasattr(data, 'file'):
                    f_name = str(data.file[b])
                else:
                    f_name = str(getattr(data[b], 'file', 'Unknown'))
                key = (k_name, f_name, int(m_id[b].item()))
                if key in frozen_top3_map:
                    topk_indices_locked[b] = frozen_top3_map[key].to(topk_indices.device)
                else:
                    frozen_top3_map[key] = topk_indices[b].detach().cpu()
                    topk_indices_locked[b] = topk_indices[b]
            topk_indices = topk_indices_locked

        # 1. 独立统计 Rank-K 性能
        for rank in range(1, k + 1):
            pred_idx = topk_indices[:, rank - 1]
            curr_errors = error_vectors.gather(1, pred_idx.unsqueeze(1)).squeeze()
            curr_ratio = (true_best_val + 1e-6) / (curr_errors + 1e-6)
            hits[rank] += (pred_idx == true_best_idx).sum().item()
            ratios[rank] += curr_ratio.sum().item()

        # 1.1 margin 分桶统计（按 top1 预测表现）
        pred_top1_idx = topk_indices[:, 0]
        pred_top1_err = error_vectors.gather(1, pred_top1_idx.unsqueeze(1)).squeeze()
        pred_top1_ratio = (true_best_val + 1e-6) / (pred_top1_err + 1e-6)

        sorted_errors, _ = torch.sort(error_vectors, dim=1, descending=False)
        rel_margin = (sorted_errors[:, 1] - sorted_errors[:, 0]) / (sorted_errors[:, 0].abs() + 1e-6)
        is_top1_hit = (pred_top1_idx == true_best_idx)

        masks = {
            "lt_1": rel_margin < 0.01,
            "1_to_3": (rel_margin >= 0.01) & (rel_margin < 0.03),
            "ge_3": rel_margin >= 0.03,
        }
        for key, mask in masks.items():
            if mask.any():
                margin_buckets[key]["count"] += int(mask.sum().item())
                margin_buckets[key]["top1_hits"] += int(is_top1_hit[mask].sum().item())
                margin_buckets[key]["top1_ratio_sum"] += float(pred_top1_ratio[mask].sum().item())

        # 2. 原始优先专家选择策略 (0 > 1,2,3 > Top-1)
        has_0 = (topk_indices == 0).any(dim=1)
        has_1 = (topk_indices == 1).any(dim=1)
        has_2 = (topk_indices == 2).any(dim=1)
        has_3 = (topk_indices == 3).any(dim=1)

        idx_priority = topk_indices[:, 0].clone()
        idx_priority = torch.where(has_3, torch.full_like(idx_priority, 3), idx_priority)
        idx_priority = torch.where(has_2, torch.full_like(idx_priority, 2), idx_priority)
        idx_priority = torch.where(has_1, torch.full_like(idx_priority, 1), idx_priority)
        idx_priority = torch.where(has_0, torch.full_like(idx_priority, 0), idx_priority)

        err_priority = error_vectors.gather(1, idx_priority.unsqueeze(1)).squeeze()
        total_hits_priority += (idx_priority == true_best_idx).sum().item()
        total_ratio_priority += ((true_best_val + 1e-6) / (err_priority + 1e-6)).sum().item()

        # 3. 指标驱动策略（Logit bias 版本）
        # 对 perf 的先验已体现在 logits_eval；这里直接采用其 Top-1 结果
        idx_exp_driven = topk_indices[:, 0].clone()

        err_exp_driven = error_vectors.gather(1, idx_exp_driven.unsqueeze(1)).squeeze()
        total_hits_expert_driven += (idx_exp_driven == true_best_idx).sum().item()
        total_ratio_expert_driven += ((true_best_val + 1e-6) / (err_exp_driven + 1e-6)).sum().item()

        # 4. 统计“随机从 Top-K 中选一个”
        topk_errors = torch.gather(error_vectors, 1, topk_indices)
        random_positions = torch.randint(0, k, (batch_size, 1), device=device)
        idx_random = topk_indices.gather(1, random_positions).squeeze()
        err_random = topk_errors.gather(1, random_positions).squeeze()
        total_hits_random_k += (idx_random == true_best_idx).sum().item()
        total_ratio_random_k += ((true_best_val + 1e-6) / (err_random + 1e-6)).sum().item()

        # 5. 统计“Top-K 最优”方案 (潜力上限)
        best_val_in_topk, best_pos_in_topk = topk_errors.min(dim=1)
        best_idx_in_topk = topk_indices.gather(1, best_pos_in_topk.unsqueeze(1)).squeeze()
        ratio_best_in_topk = (true_best_val + 1e-6) / (best_val_in_topk + 1e-6)
        total_hits_best_in_topk += (best_idx_in_topk == true_best_idx).sum().item()
        total_ratio_best_in_topk += ratio_best_in_topk.sum().item()

        for i in range(5):
            mask = (m_id == i)
            if mask.any():
                m_ratios[i].extend(ratio_best_in_topk[mask].tolist())

        # 抽样打印
        if print_sample_count < 5:
            for b in range(min(batch_size, 1)):
                m_name = metric_map.get(int(m_id[b].item()), "Unknown")
                p_top3 = [expert_names[idx] for idx in topk_indices[b].tolist()]
                t_top3 = [expert_names[idx] for idx in true_topk_indices[b, :k].tolist()]
                final_choice = expert_names[idx_exp_driven[b].item()]
                print(f"\n[决策对比] 指标: {m_name} | 推荐前3: {p_top3} | 实际最优前3: {t_top3}")
                print(f"  > 专家驱动策略最终选择: {final_choice}")
                print("-" * 45)
                print_sample_count += 1

    # --- 最终打印报表 ---
    print(f"\n  [Validation Metrics - Samples: {total_count}]")
    if metric_name is not None:
        print(f"  [Validation Metric Scope] {metric_name}")
    for r in range(1, k + 1):
        print(f"  Rank-{r} Individual  : Acc={hits[r] / total_count:.1%}, Ratio={ratios[r] / total_count:.4f}")

    print(f"  " + "-" * 45)
    print(
        f"  Priority (0>123>T1): Acc={total_hits_priority / total_count:.1%}, Ratio={total_ratio_priority / total_count:.4f}")
    print(
        f"  Expert-Driven (New): Acc={total_hits_expert_driven / total_count:.1%}, Ratio={total_ratio_expert_driven / total_count:.4f}")
    print(
        f"  Random-from-Top-{k} : Acc={total_hits_random_k / total_count:.1%}, Ratio={total_ratio_random_k / total_count:.4f}")
    print(
        f"  Best-in-Top-{k} (Pot): Acc={total_hits_best_in_topk / total_count:.1%}, Ratio={total_ratio_best_in_topk / total_count:.4f}")
    print(f"  " + "-" * 45)
    print("  Margin Buckets (Top-1 Diagnostics):")
    bucket_labels = {
        "lt_1": "margin < 1%",
        "1_to_3": "1% <= margin < 3%",
        "ge_3": "margin >= 3%",
    }
    for key in ["lt_1", "1_to_3", "ge_3"]:
        c = margin_buckets[key]["count"]
        if c > 0:
            b_acc = margin_buckets[key]["top1_hits"] / c
            b_ratio = margin_buckets[key]["top1_ratio_sum"] / c
        else:
            b_acc = 0.0
            b_ratio = 0.0
        print(f"    - {bucket_labels[key]:<18} | Count={c:<6d} | Top1 Acc={b_acc:.1%} | Top1 Ratio={b_ratio:.4f}")

    m_avg_ratios = {i: (np.mean(m_ratios[i]) if m_ratios[i] else 0) for i in range(5)}

    top1_acc = hits[1] / total_count if total_count > 0 else 0.0
    top1_ratio = ratios[1] / total_count if total_count > 0 else 0.0
    rank_avg_ratios = {r: (ratios[r] / total_count if total_count > 0 else 0.0) for r in range(1, k + 1)}
    best_rank = max(rank_avg_ratios, key=rank_avg_ratios.get)
    best_rank_ratio = rank_avg_ratios[best_rank]
    topk_best_acc = total_hits_best_in_topk / total_count if total_count > 0 else 0.0
    topk_best_ratio = total_ratio_best_in_topk / total_count if total_count > 0 else 0.0
    return top1_acc, top1_ratio, topk_best_acc, topk_best_ratio, m_avg_ratios, best_rank, best_rank_ratio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train selector in multi-metric or single-metric mode.")
    parser.add_argument(
        "--target_m_id",
        type=int,
        default=CONFIG_TARGET_M_ID,
        choices=[0, 1, 2, 3, 4],
        help="Train/eval only one metric id (0:perf,1:lut,2:ff,3:dsp,4:bram). Default comes from src/config.py target_m_id.",
    )
    parser.add_argument(
        "--train_skip_all_zero_config",
        type=int,
        default=1 if CONFIG_TRAIN_SKIP_ALL_ZERO else 0,
        choices=[0, 1],
        help="Whether to filter configs whose 5 true metrics are all 0 in TRAIN set (1: yes, 0: no).",
    )
    parser.add_argument(
        "--test_skip_all_zero_config",
        type=int,
        default=1 if CONFIG_TEST_SKIP_ALL_ZERO else 0,
        choices=[0, 1],
        help="Whether to filter configs whose 5 true metrics are all 0 in VAL/TEST set (1: yes, 0: no).",
    )
    args = parser.parse_args()

    target_m_id = args.target_m_id
    train_skip_all_zero_config = bool(args.train_skip_all_zero_config)
    test_skip_all_zero_config = bool(args.test_skip_all_zero_config)
    target_metric_name = TARGET_NAMES[target_m_id] if target_m_id is not None else "all"
    print(f">>> Target metric mode: {target_metric_name} (target_m_id={target_m_id})")
    print(f">>> Train all-zero filter: {train_skip_all_zero_config}")
    print(f">>> Val/Test all-zero filter: {test_skip_all_zero_config}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_DIR = os.path.join(BASE_DIR, "save_models_and_data")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 加载标签数据
    label_path = os.path.join(BASE_DIR, "algorithm_select/labels.json")
    label_dict = load_labels_from_json(label_path)

    from make_labels import get_aligned_data

    all_aligned_info, _ = get_aligned_data()

    # full_dataset = AlgorithmSelectDataset(all_aligned_info, label_dict, extract_hybrid_meta)   # 以kernel为样本才需要加入，以配置为样本不需要

    # 2. 划分数据集 (沿用你之前的划分逻辑)
    train_list, val_list = get_train_val_test_sets(all_aligned_info)  # 以配置为样本使用这个
    # train_samples, val_samples = get_train_val_test_sets(full_dataset.samples)    # 以kernel为样本使用这个

    # 3. 初始化 DataLoader
    # 注意：此处使用的 AlgorithmSelectDataset 应当是你修改后的“样本展开版”
    # train_dataset = AlgorithmSelectDataset(train_list, label_dict, extract_hybrid_meta)   # 这几行都是以配置为样本的
    # val_dataset = AlgorithmSelectDataset(val_list, label_dict, extract_hybrid_meta)

    train_dataset = AlgorithmSelectDataset(
        train_list, label_dict, extract_hybrid_meta,
        target_m_id=target_m_id,
        skip_all_zero_config=train_skip_all_zero_config,
    )      # 三特征融合使用这个
    val_dataset = AlgorithmSelectDataset(
        val_list, label_dict, extract_hybrid_meta,
        target_m_id=target_m_id,
        skip_all_zero_config=test_skip_all_zero_config,
    )

    print(f">>> Dataset size after labels alignment: train={len(train_dataset)}, val={len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("对齐 labels.json 后训练集或验证集为空，请检查 labels.json 与过滤口径。")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # train_dataset = AlgorithmSelectDataset(None, None, extract_hybrid_meta, pre_aggregated_samples=train_samples) # 这几行注释都是以kernel为样本的
    # val_dataset = AlgorithmSelectDataset(None, None, extract_hybrid_meta, pre_aggregated_samples=val_samples)
    # if len(train_dataset) == 0:
    #     raise ValueError("训练集为空，请检查 Kernel 名称划分逻辑！")
    # # 5. 修改 DataLoader 参数 (Kernel 级样本少，Batch Size 调小)
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 4. 初始化模型和优化器
    model = AlgorithmSelector(
        gcn_params={'hidden_channels': 128},
        # meta_base_dim=9,
        # ast_tail_dim=16,  # 如果是以配置为样本就需要
        num_tools=7
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # --- 【新增：断点续训逻辑】 ---
    checkpoint_name = f"best_ppo_model_{target_metric_name}.pth"
    checkpoint_path = os.path.join(SAVE_DIR, checkpoint_name)
    best_ratio = 0.0

    if os.path.exists(checkpoint_path):
        print(f"\n>>> 检测到历史模型权重: {checkpoint_path}")
        try:
            # 加载权重
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            print(">>> 成功加载权重！正在通过验证集初始化最佳 Ratio...")

            # 运行一次验证以同步 best_ratio，防止被低分模型覆盖
            initial_top1_acc, initial_top1_ratio, initial_topk_acc, initial_topk_ratio, m_ratios, initial_best_rank, initial_best_rank_ratio = evaluate_performance(
                model, val_loader, device, metric_name=target_metric_name,
                lock_top3_for_eval=phase2_active, frozen_top3_map=frozen_eval_top3_map)
            best_ratio = initial_best_rank_ratio

            if target_m_id is not None:
                detail_str = f"{target_metric_name.upper()}: {m_ratios[target_m_id]:.4f}"
            else:
                detail_str = " | ".join([f"{TARGET_NAMES[i].upper()}: {m_ratios[i]:.4f}" for i in range(5)])
            print(f">>> 初始最佳 Rank Ratio 已同步为: Rank-{initial_best_rank}={best_ratio:.4f} ({detail_str})")
            print(f">>> 参考 TopK 上限: Acc={initial_topk_acc:.1%}, Ratio={initial_topk_ratio:.4f}")
        except Exception as e:
            print(f">>> 加载模型失败，原因: {e}。将从头开始训练。")
            best_ratio = 0.0
    else:
        print("\n>>> 未检测到历史模型，开始全新训练。")
    # -----------------------------

    print("\n>>> 开始 PPO 训练 (Configuration-level 样本)...")

    # --- 二阶段训练配置 ---
    freeze_epoch = 50
    phase2_trigger_top3_ratio = 0.85
    phase2_active = False
    teacher_model = None
    frozen_eval_top3_map = {}

    for epoch in range(1000):
        # 熵权重衰减逻辑
        ent_w = max(0.005, 0.02 * (0.995 ** epoch))

        # 训练一个 Epoch
        # train_reward, train_acc, train_ratio = train_ppo_style(model, train_loader, optimizer, device, ent_w)
        train_reward, train_acc, train_ratio = train_ppo_style(
            model, train_loader, optimizer, device, ent_w, epoch,
            phase2_active=phase2_active, teacher_model=teacher_model
        )

        # 验证
        val_top1_acc, val_top1_ratio, val_topk_acc, val_topk_ratio, m_ratios, val_best_rank, val_best_rank_ratio = evaluate_performance(
            model, val_loader, device, metric_name=target_metric_name,
            lock_top3_for_eval=phase2_active, frozen_top3_map=frozen_eval_top3_map
        )

        # 打印日志
        is_best = val_best_rank_ratio > best_ratio

        print(
            f"Epoch {epoch + 1:04d} | "
            f"Train Top1 Acc/Ratio: {train_acc:.1%}/{train_ratio:.4f} | "
            f"Val Top1 Acc/Ratio: {val_top1_acc:.1%}/{val_top1_ratio:.4f} | "
            f"Val Best-in-Top3 Acc/Ratio: {val_topk_acc:.1%}/{val_topk_ratio:.4f} | "
            f"Val BestRank Ratio: Rank-{val_best_rank}/{val_best_rank_ratio:.4f} | "
            f"Phase2={'ON' if phase2_active else 'OFF'}"
        )

        # 二阶段触发：达到固定轮次，或 Best-in-Top-3 ratio 提前达到阈值
        if (not phase2_active) and (((epoch + 1) >= freeze_epoch) or (val_topk_ratio >= phase2_trigger_top3_ratio)):
            phase2_active = True
            teacher_model = copy.deepcopy(model).to(device)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            frozen_eval_top3_map.clear()
            trigger_reason = (
                f"epoch>={freeze_epoch}" if (epoch + 1) >= freeze_epoch
                else f"val_topk_ratio>={phase2_trigger_top3_ratio:.2f}"
            )
            print(f">>> [Phase-2 Activated] reason={trigger_reason}, epoch={epoch + 1}, "
                  f"val_topk_ratio={val_topk_ratio:.4f}")

        # 如果效果更好，保存模型
        if is_best:
            best_ratio = val_best_rank_ratio
            torch.save(model.state_dict(), checkpoint_path)
            if target_m_id is not None:
                detail_str = f"{target_metric_name.upper()}_Ratio: {m_ratios[target_m_id]:.4f}"
            else:
                detail_str = " | ".join([f"{TARGET_NAMES[i].upper()}_Ratio: {m_ratios[i]:.4f}" for i in range(5)])
            print(f"   [!] 刷新纪录并已保存模型 (BestRank=Rank-{val_best_rank}, Ratio={val_best_rank_ratio:.4f}) -> {detail_str}")

            # print(f"   [*] 正在执行 Kernel 级别的物理误差分析 (Top-3 择优模式)...")
            # calculate_pipeline_rmse(
            #     selector=model,
            #     loader=val_loader,
            #     device=device,
            #     target_names=TARGET_NAMES,
            #     k=3
            # )

    print("\n训练结束。")

