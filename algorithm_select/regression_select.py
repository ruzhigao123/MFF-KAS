

import torch
import torch.nn as nn
import os
import sys
import builtins
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from scipy.stats import kendalltau

# --- 1. 基础配置与路径 ---
BASE_DIR = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master"
sys.path.append(BASE_DIR)

from src.model import Net
from src.utils import MLP as MLP_Class, _get_y_with_target
from make_labels import get_aligned_data, MockFlags, get_feat

# 数据集基础定义
MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'doitgen',
               'mvt', 'fdtd-2d', 'gemver', 'gemm-p', 'gesummv',
               'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d']

# 专家模型配置
EXPERT_CONFIGS = [
    {"id": 0, "mode": "all", "file": "all_regression_model_state_dict.pth", "dim": 153},
    {"id": 1, "mode": "ast_cdfg", "file": "ast_and_cdfg_regression_model_state_dict.pth", "dim": 153},
    {"id": 2, "mode": "cdfg_text", "file": "cdfg_and_text_regression_model_state_dict.pth", "dim": 153},
    {"id": 3, "mode": "ast_text", "file": "ast_and_text_regression_model_state_dict.pth", "dim": 153},
    {"id": 4, "mode": "cdfg", "file": "cdfg_regression_model_state_dict.pth", "dim": 153},
    {"id": 5, "mode": "ast", "file": "ast_regression_model_state_dict.pth", "dim": 784},
    {"id": 6, "mode": "text", "file": "text_regression_model_state_dict.pth", "dim": 153},
]

FUSION_MODES = ['all', 'ast_cdfg', 'cdfg_text', 'ast_text']


# --- 2. 核心：三特征融合模块 ---
class TripleModalFusionBlock(nn.Module):
    def __init__(self, g_dim=128, s_dim=768, a_dim=784, embed_dim=256, num_heads=8, dropout=0.1):
        super(TripleModalFusionBlock, self).__init__()
        # 1. 维度对齐层
        self.proj_g = nn.Linear(g_dim, embed_dim)
        self.proj_s = nn.Linear(s_dim, embed_dim)
        self.proj_a = nn.Linear(a_dim, embed_dim)

        # 2. 第一阶段交叉注意力
        self.attn_gs = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        # 3. 第二阶段交叉注意力
        self.attn_gsa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 4. 门控融合层
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, 1), nn.Sigmoid())

    def forward(self, h_g, h_s, h_a):
        g_emb = self.proj_g(h_g).unsqueeze(1)
        s_emb = self.proj_s(h_s).unsqueeze(1)
        a_emb = self.proj_a(h_a).unsqueeze(1)

        # 第一阶段：CDFG + Text
        h_gs, _ = self.attn_gs(query=g_emb, key=s_emb, value=s_emb)
        h_gs = self.norm1(h_gs + g_emb)

        # 第二阶段：(CDFG+Text) + AST
        h_fuse, _ = self.attn_gsa(query=h_gs, key=a_emb, value=a_emb)
        h_fuse = self.norm2(h_fuse + h_gs)
        h_fuse = h_fuse.squeeze(1)

        # 第三阶段：门控控制
        g_proj = g_emb.squeeze(1)
        gate_val = self.gate(torch.cat([g_proj, h_fuse], dim=-1))
        h_final = gate_val * g_proj + (1 - gate_val) * h_fuse
        return h_final


# --- 3. 辅助函数 ---
def enforce_dim(tensor, target_dim):
    """强制对齐特征维度，解决特征提取不一致导致的网络崩溃"""
    if tensor.shape[1] == target_dim:
        return tensor
    elif tensor.shape[1] < target_dim:
        pad = torch.zeros((tensor.shape[0], target_dim - tensor.shape[1]), device=tensor.device)
        return torch.cat([tensor, pad], dim=1)
    else:
        return tensor[:, :target_dim]


def compute_metrics(y_true, y_pred):
    """计算包含 Tau 在内的所有指标"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100
    tau, _ = kendalltau(y_true, y_pred)
    # 处理 Tau 可能返回 NaN 的情况（当序列完全一致或无法排序时）
    if np.isnan(tau): tau = 0.0
    return mse, rmse, mae, mape, tau


# --- 4. 主干推理流程 ---
def run_regression_select():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EXP_DIR = os.path.join(BASE_DIR, "model_select_weight")
    target_list = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']

    # 1. 严格过滤测试集
    all_aligned = get_aligned_data()[0]
    seen_mach = set(MACHSUITE_KERNEL[:5])
    seen_poly = set(poly_KERNEL[:10])
    test_list = [info for info in all_aligned if info['key'][1] not in seen_mach and info['key'][1] not in seen_poly]

    if not test_list:
        print("错误：测试集为空，请检查数据。")
        sys.exit()

    print(
        f">>> 启动 Oracle 评估 | 训练集覆盖 Kernel: {len(seen_mach) + len(seen_poly)} 个 | 测试集 Kernel: {len(set(info['key'][1] for info in test_list))} 个")

    # 存储所有专家在所有样本上的预测结果: expert_preds[exp_id][kname][m_idx] = [(true, pred), ...]
    expert_preds = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    print("\n>>> [1/2] 提取特征并评估 7 个专家...")
    for exp_cfg in EXPERT_CONFIGS:
        exp_id, mode = exp_cfg['id'], exp_cfg['mode']
        builtins.FLAGS = MockFlags(mode)

        pth_path = os.path.join(EXP_DIR, exp_cfg['file'])
        if not os.path.exists(pth_path): continue

        # 初始化模型并注入强制正确的 FusionBlock
        model = Net(in_channels=exp_cfg['dim'], target_list=target_list).to(device)
        if hasattr(model, 'fusion_block'):
            model.fusion_block = TripleModalFusionBlock().to(device)

        ckpt = torch.load(pth_path, map_location=device)

        # 适配 MLP
        exp_dim = ckpt["MLPs.perf.layers.0.weight"].shape[1] if "MLPs.perf.layers.0.weight" in ckpt else 128
        if exp_dim != model.MLPs['perf'].layers[0].in_features:
            for name in target_list:
                model.MLPs[name] = MLP_Class(exp_dim, 1, 'relu', 2, [128, 128]).to(device)

        model.load_state_dict(ckpt, strict=False)
        model.eval()

        with torch.no_grad():
            for info in test_list:
                kname = info['key'][1]
                d_cdfg = torch.load(info['cdfg'], map_location=device)
                d_text = torch.load(info['text'], map_location=device)
                d_ast = torch.load(info['ast'], map_location=device)

                # A. 骨干图特征 (128维)
                batch_obj = d_ast if mode == 'ast' else d_cdfg
                h_g = model.gnn_model(Batch.from_data_list([batch_obj]).to(device))

                # B. 辅助特征与模态屏蔽
                if mode in FUSION_MODES and hasattr(model, 'fusion_block'):
                    # 提取并池化
                    t_f = get_feat(d_text).to(device)
                    a_f = get_feat(d_ast).to(device)
                    h_s = global_mean_pool(t_f, torch.zeros(t_f.size(0), dtype=torch.long, device=device))
                    h_a = global_mean_pool(a_f, torch.zeros(a_f.size(0), dtype=torch.long, device=device))

                    # 强制对齐维度 (768 和 784)
                    h_s = enforce_dim(h_s, 768)
                    h_a = enforce_dim(h_a, 784)

                    # 严格依照训练时的逻辑屏蔽缺失模态
                    if mode == 'ast_text':
                        h_g = torch.zeros_like(h_g)
                    elif mode == 'ast_cdfg':
                        h_s = torch.zeros_like(h_s)
                    elif mode == 'cdfg_text':
                        h_a = torch.zeros_like(h_a)

                    input_feat = model.fusion_block(h_g, h_s, h_a)
                else:
                    input_feat = h_g

                # C. 记录预测值
                for m_idx, m_name in enumerate(target_list):
                    true_val = _get_y_with_target(d_cdfg, m_name).item()
                    pred_val = model.MLPs[m_name](input_feat).item()
                    expert_preds[exp_id][kname][m_idx].append((true_val, pred_val))

        del model
        torch.cuda.empty_cache()

    print("\n>>> [2/2] 生成 Oracle 最优决策报表...")
    print("\n" + "=" * 95)
    print(f"{'指标':<10} | {'Kernel 名称':<18} | {'最优专家':<15} | {'该专家 MSE':<12}")
    print("-" * 95)

    breakdown_loss = {}

    # 按照 Target 输出
    for m_idx, m_name in enumerate(target_list):
        m_all_trues, m_all_preds = [], []

        for kname in sorted(set(info['key'][1] for info in test_list)):
            best_exp_id, min_mse = -1, float('inf')
            best_pairs = []

            # 寻找该 Kernel 该指标的最优专家
            for eid in range(7):
                pairs = expert_preds[eid][kname][m_idx]
                if not pairs: continue
                mse = np.mean([(p[0] - p[1]) ** 2 for p in pairs])
                if mse < min_mse:
                    min_mse = mse
                    best_exp_id = eid
                    best_pairs = pairs

            m_all_trues.extend([p[0] for p in best_pairs])
            m_all_preds.extend([p[1] for p in best_pairs])

            print(
                f"{m_name:<10} | {kname:<18} | ID:{best_exp_id:<2} ({EXPERT_CONFIGS[best_exp_id]['mode']:<8}) | {min_mse:.4e}")

        # 计算并打印该指标的总统计 (RMSE, MAE, MAPE, Tau)
        m_mse, m_rmse, m_mae, m_mape, m_tau = compute_metrics(m_all_trues, m_all_preds)
        print(
            f"--- Target: {m_name:<8} | RMSE: {m_rmse:.4e} | MAE: {m_mae:.4e} | MAPE: {m_mape:.4f} | Tau: {m_tau:.4f}")
        print("-" * 95)

        breakdown_loss[m_name] = m_mse

    # 5. 打印最后的总结论，格式完全匹配你的需求
    print("\nInference loss breakdown:")
    # 为了跟你的格式一样，转成漂亮的字典字符串
    formatted_breakdown = {k: float(v) for k, v in breakdown_loss.items()}
    print(formatted_breakdown)

    overall_loss = sum(breakdown_loss.values()) / len(breakdown_loss)
    print(f"Overall Inference loss: {overall_loss:.7f}")


if __name__ == "__main__":
    run_regression_select()