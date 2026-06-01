
import torch
import os
import json
import sys
import re
import glob
from tqdm import tqdm
import numpy as np
import builtins
from collections import defaultdict
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

# --- 1. 路径配置 ---
CDFG_ROOT = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_dataset/graph"
TEXT_ROOT = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_dataset/code"
AST_ROOT = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_datasets/code/ast_final_dataset_1"
# 权重目录指向包含 all_regression_model_state_dict.pth 等文件的目录
WEIGHT_DIR = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/model_select_weight_del_0"
SAVE_PATH = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/algorithm_select/labels.json"
SKIP_ALL_ZERO_CONFIG = True

sys.path.append("/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master")
from src.model import Net
from src.utils import _get_y_with_target, MLP as MLP_Class


def get_aligned_data():
    print("正在建立三模态对齐索引...")

    def build_index(root, pattern, regex):
        idx = {}
        for f in glob.glob(os.path.join(root, "**", pattern), recursive=True):
            p = f.split(os.sep)
            m = re.search(regex, p[-1])
            if m: idx[(p[-3], p[-2], int(m.group()))] = f
        return idx

    cdfg_idx = build_index(CDFG_ROOT, "data_*.pt", r"\d+")
    text_idx = build_index(TEXT_ROOT, "*.pt", r"\d+")
    ast_idx = build_index(AST_ROOT, "*.ast.pt", r"\d+")
    common_keys = [k for k in cdfg_idx if (k[0], k[1], k[2] + 1) in text_idx and k in ast_idx]
    aligned_data = []
    for k in common_keys:
        aligned_data.append(
            {'key': k, 'cdfg': cdfg_idx[k], 'text': text_idx[(k[0], k[1], k[2] + 1)], 'ast': ast_idx[k]})
    return aligned_data, cdfg_idx


class MockFlags:
    def __init__(self, mode):
        self.ablation_stu = mode
        self.task = 'regression'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = 'gcn'
        self.comparative_if = True
        self.gnn_model_type = 'gcn'
        self.hidden_num = 128
        self.out_dim = 1
        self.num_features = 153
        self.dropout = 0.1
        self.layer_norm = True
        self.skip = True
        self.encode_log = False
        self.normalizer = 1.0
        # self.comparative_model = 'pna'
        self.target = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']


def get_feat(obj):
    if hasattr(obj, 'x'): return obj.x
    return obj


def _is_all_zero_config(d_cdfg, eps: float = 1e-12) -> bool:
    vals = []
    for name in ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']:
        y = _get_y_with_target(d_cdfg, name).view(-1)
        vals.append(float(y.mean().item()) if y.numel() > 0 else 0.0)
    return all(abs(v) <= eps for v in vals)


def make_labels():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aligned_data, cdfg_idx_map = get_aligned_data()
    if SKIP_ALL_ZERO_CONFIG:
        before_cnt = len(aligned_data)
        filtered = []
        for info in tqdm(aligned_data, desc="Filtering all-zero configs"):
            try:
                d_cdfg = torch.load(info['cdfg'], map_location='cpu')
                if not _is_all_zero_config(d_cdfg):
                    filtered.append(info)
            except Exception:
                continue
        aligned_data = filtered
        print(f">>> [make_labels filter] before={before_cnt}, removed={before_cnt - len(aligned_data)}, remain={len(aligned_data)}")
    common_keys = [item['key'] for item in aligned_data]

    # --- 核心对齐修改：使用与 inference_engine.py 一致的专家配置 ---
    tools = [
        {"id": 0, "file": "all_regression_model_state_dict.pth", "mode": "all", "dim": 153},
        {"id": 1, "file": "ast_and_cdfg_regression_model_state_dict.pth", "mode": "ast_cdfg", "dim": 153},
        {"id": 2, "file": "ast_and_text_regression_model_state_dict.pth", "mode": "ast_text", "dim": 153},
        {"id": 3, "file": "cdfg_and_text_regression_model_state_dict.pth", "mode": "cdfg_text", "dim": 153},
        {"id": 4, "file": "cdfg_regression_model_state_dict.pth", "mode": "cdfg_only", "dim": 153},
        {"id": 5, "file": "ast_regression_model_state_dict.pth", "mode": "ast_only", "dim": 784},
        {"id": 6, "file": "text_regression_model_state_dict.pth", "mode": "LM", "dim": 153}
    ]

    target_names = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']
    # 存储所有样本在所有专家下的误差: [Tool_ID, Sample_Idx, Metric_Idx]
    all_errors = np.zeros((7, len(common_keys), 5))

    # 第一阶段：循环专家列表进行推理
    for t in tools:
        mode_id = t['id']
        mode_name = t['mode']
        weight_path = os.path.join(WEIGHT_DIR, t['file'])

        if not os.path.exists(weight_path):
            print(f"⚠️ 警告: 找不到专家权重文件 {weight_path}，跳过...")
            continue

        print(f"\n>>> [1/2] 正在评估专家 {mode_id} ({mode_name}) 使用权重: {t['file']}")

        builtins.FLAGS = MockFlags(mode_name)
        model = Net(in_channels=t['dim'], target_list=target_names).to(device)
        ckpt = torch.load(weight_path, map_location='cpu')

        # --- 处理 MLP1 与 MLPs 的结构转换 (兼容老版本权重) ---
        if "MLP1.0.weight" in ckpt:
            new_ckpt = {k: v for k, v in ckpt.items() if "MLP1" not in k}
            for t_idx, t_name in enumerate(target_names):
                new_ckpt[f"MLPs.{t_name}.layers.0.weight"] = ckpt["MLP1.0.weight"]
                new_ckpt[f"MLPs.{t_name}.layers.0.bias"] = ckpt["MLP1.0.bias"]
                new_ckpt[f"MLPs.{t_name}.layers.2.weight"] = ckpt["MLP1.2.weight"][t_idx:t_idx + 1, :]
                new_ckpt[f"MLPs.{t_name}.layers.2.bias"] = ckpt["MLP1.2.bias"][t_idx:t_idx + 1]
            ckpt = new_ckpt

        # --- 自适应调整 MLP 接收维度 (256 vs 128) ---
        exp_dim = ckpt["MLPs.perf.layers.0.weight"].shape[1]
        if exp_dim != model.MLPs['perf'].layers[0].in_features:
            for name in target_names:
                model.MLPs[name] = MLP_Class(exp_dim, 1, 'relu', 2, [128, 128]).to(device)

        model.load_state_dict(ckpt, strict=False)
        model.eval()

        with torch.no_grad():
            for s_idx, info in enumerate(tqdm(aligned_data, desc=f"Tool {mode_id} Inference")):
                d_cdfg = torch.load(info['cdfg'], map_location='cpu')
                t_f_raw = torch.load(info['text'], map_location='cpu')
                a_obj_raw = torch.load(info['ast'], map_location='cpu')

                current_batch_obj = Batch.from_data_list([a_obj_raw if mode_name == 'ast_only' else d_cdfg]).to(device)
                t_f, a_f = get_feat(t_f_raw).to(device), get_feat(a_obj_raw).to(device)

                h_g = model.gnn_model(current_batch_obj)
                h_s = global_mean_pool(t_f, torch.zeros(t_f.size(0), dtype=torch.long, device=device))
                h_a = global_mean_pool(a_f, torch.zeros(a_f.size(0), dtype=torch.long, device=device))

                # --- 推理时的特征选择逻辑 ---
                if exp_dim == 256:
                    if mode_name == 'ast_text':
                        h_g = torch.zeros_like(h_g)
                    elif mode_name == 'ast_cdfg':
                        h_s = torch.zeros_like(h_s)
                    elif mode_name == 'cdfg_text':
                        h_a = torch.zeros_like(h_a)
                    input_to_mlp = model.fusion_block(h_g=h_g, h_s=h_s, h_a=h_a)
                else:
                    if mode_name == 'LM':
                        input_to_mlp = h_s
                    elif mode_name == 'ast_only':
                        input_to_mlp = h_g
                    else:
                        input_to_mlp = h_g

                for m_idx, name in enumerate(target_names):
                    pred = model.MLPs[name](input_to_mlp)
                    actual = _get_y_with_target(d_cdfg, name).view(-1, 1).to(device)
                    # 存储绝对误差平均值
                    all_errors[mode_id, s_idx, m_idx] = torch.abs(pred - actual).mean().item()

        del model
        torch.cuda.empty_cache()

    # 第二阶段：配置级 (Configuration-level) 标签生成 (软标签模式)
    print("\n>>> [2/2] 正在生成基于配置 (Configuration-level) 的误差分布特征 (软标签)...")

    label_dict = {}
    for s_idx, key in enumerate(common_keys):
        kname = key[1]
        fname = os.path.basename(cdfg_idx_map[key])
        for m_idx in range(5):
            # 获取该样本在特定指标下，7个专家的所有预测误差向量
            expert_loss_vector = [round(float(err), 6) for err in all_errors[:, s_idx, m_idx]]

            # 保存的是一个长为 7 的误差数组，用于强化学习或回归型选择器的训练
            label_dict[str((kname, fname, m_idx))] = expert_loss_vector

    # 保存结果
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, 'w') as f:
        json.dump(label_dict, f, indent=4)

    print(f"✅ 误差分布特征生成完成！")
    print(f"📊 统计：共处理 {len(common_keys)} 个配置样本，生成 {len(label_dict)} 条 7维误差向量。")
    print(f"💾 结果已保存至: {SAVE_PATH}")


if __name__ == "__main__":
    make_labels()


