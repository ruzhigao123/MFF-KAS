
import torch
from torch.utils.data import Dataset
import os
import json
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.utils import _get_y_with_target


class AlgorithmSelectDataset(Dataset):
    def __init__(
        self,
        raw_data_list,
        label_dict,
        extract_func,
        target_m_id=None,
        skip_all_zero_config=True,
        aggregate_by_kernel=False,
    ):
        super(AlgorithmSelectDataset, self).__init__()
        self.label_dict = label_dict
        self.samples = []
        self.extract_func = extract_func
        self.target_m_id = target_m_id
        self.skip_all_zero_config = skip_all_zero_config
        self.aggregate_by_kernel = aggregate_by_kernel
        if aggregate_by_kernel:
            self.prepare_kernel_aggregated_samples(raw_data_list)
        else:
            self.prepare_flattened_samples(raw_data_list)

    def prepare_kernel_aggregated_samples(self, raw_data_list):
        """
        每个 (kernel_name, m_id) 一条样本：7 维误差 = 该 kernel 下所有（有标签的）配置误差的逐专家平均。
        图 / text / ast 使用「代表配置」一条（按配置编号中位数对应的 entry）。
        meta：仅用 extract 结果的前 9 维（AST JSON 侧），在配置上平均；后 16 维（来自 ast.pt 尾部）不用，填 0，整体仍为 [25] 供下游切片 meta_base/meta_tail。
        """
        from collections import defaultdict

        print("\n[Kernel 聚合] 样本粒度 = 每个 kernel × 指标；标签为配置级误差向量在配置维度上 mean(...)")
        print("[Kernel 聚合] meta：仅 9 维 JSON 特征（跨配置均值），尾 16 维置 0（不使用 ast.pt 硬件尾特征）")

        target_names = ["perf", "util-LUT", "util-FF", "util-DSP", "util-BRAM"]
        entry_by_kc = {}
        accum = defaultdict(lambda: {"c_idxs": [], "errs": []})
        checked_configs = 0
        skipped_zero_configs = 0
        skipped_missing_label = 0

        def _ast_path_from_entry(entry):
            raw_ast_path = entry.get("ast_raw_path") or entry.get("ast_path") or entry.get("ast")
            if raw_ast_path is None and "cdfg" in entry:
                raw_ast_path = str(entry["cdfg"])
            return raw_ast_path

        for entry in tqdm(raw_data_list, desc="Kernel 聚合·扫描配置"):
            _, k_name, c_idx = entry["key"]
            checked_configs += 1

            if self.skip_all_zero_config:
                try:
                    d_cdfg = torch.load(entry["cdfg"], map_location="cpu")
                    true_vals = []
                    for t_name in target_names:
                        y = _get_y_with_target(d_cdfg, t_name).view(-1)
                        true_vals.append(float(y.mean().item()) if y.numel() > 0 else 0.0)
                    if all(abs(v) <= 1e-12 for v in true_vals):
                        skipped_zero_configs += 1
                        continue
                except Exception:
                    pass

            entry_by_kc[(k_name, c_idx)] = entry

            metric_ids = [self.target_m_id] if self.target_m_id is not None else range(5)
            for m_id in metric_ids:
                label_key = (k_name, f"data_{c_idx}.pt", m_id)
                if label_key not in self.label_dict:
                    skipped_missing_label += 1
                    continue
                vec = np.asarray(self.label_dict[label_key], dtype=np.float64)
                accum[(k_name, m_id)]["c_idxs"].append(c_idx)
                accum[(k_name, m_id)]["errs"].append(vec)

        if self.skip_all_zero_config:
            print(
                f"[过滤统计] 配置扫描: {checked_configs}, 全0跳过: {skipped_zero_configs}, "
                f"缺失label计数(槽位): {skipped_missing_label}"
            )

        for (k_name, m_id), blob in accum.items():
            pairs = list(zip(blob["c_idxs"], blob["errs"]))
            by_c = {}
            for c_idx, err_vec in pairs:
                if c_idx not in by_c:
                    by_c[c_idx] = err_vec
            c_sorted = sorted(by_c.keys())
            if not c_sorted:
                continue
            errs_stack = np.stack([by_c[c] for c in c_sorted], axis=0)
            avg_err = errs_stack.mean(axis=0)

            repr_c = c_sorted[len(c_sorted) // 2]
            repr_entry = entry_by_kc.get((k_name, repr_c))
            if repr_entry is None:
                continue

            meta9_list = []
            for cc in c_sorted:
                e = entry_by_kc[(k_name, cc)]
                m = self.extract_func(_ast_path_from_entry(e))
                if not isinstance(m, torch.Tensor):
                    m = torch.as_tensor(m, dtype=torch.float32)
                v = m.float().flatten()
                if v.numel() >= 9:
                    meta9_list.append(v[:9])
                else:
                    pad = torch.zeros(9, dtype=v.dtype, device=v.device)
                    pad[: v.numel()] = v
                    meta9_list.append(pad)
            meta9_mean = torch.stack(meta9_list, dim=0).mean(dim=0)
            tail16 = torch.zeros(16, dtype=meta9_mean.dtype)
            meta_feat = torch.cat([meta9_mean, tail16], dim=0)

            raw_ast_repr = _ast_path_from_entry(repr_entry)
            self.samples.append(
                {
                    "info": repr_entry,
                    "kernel_name": k_name,
                    "m_id": m_id,
                    "label": torch.tensor(avg_err, dtype=torch.float32),
                    "meta_feat": meta_feat,
                    "ast_path": raw_ast_repr,
                }
            )

        if self.target_m_id is not None:
            print(f"[单指标模式] target_m_id={self.target_m_id}, target_name={target_names[self.target_m_id]}")
        print(f"[完成] Kernel 级样本数: {len(self.samples)}（期望≈ 训练/验证 kernel 数 × 指标数）")
        if len(self.samples) == 0:
            raise ValueError("Kernel 聚合后样本为空：请检查 labels 与 aligned 列表。")

    def prepare_flattened_samples(self, raw_data_list):
        print(f"\n[数据展开] 正在匹配标签并提取路径...")

        valid_count = 0
        skipped_zero_configs = 0
        checked_configs = 0
        candidate_metric_slots = 0
        skipped_missing_label = 0
        target_names = ["perf", "util-LUT", "util-FF", "util-DSP", "util-BRAM"]
        for entry in tqdm(raw_data_list, desc="对齐标签"):
            # entry['key'] 结构通常为 [dataset_name, kernel_name, config_idx]
            dataset_name, k_name, c_idx = entry['key']
            checked_configs += 1

            # 可选配置级过滤：若5个真实指标全为0，则整配置跳过
            if self.skip_all_zero_config:
                try:
                    d_cdfg = torch.load(entry['cdfg'], map_location='cpu')
                    true_vals = []
                    for t_name in target_names:
                        y = _get_y_with_target(d_cdfg, t_name).view(-1)
                        true_vals.append(float(y.mean().item()) if y.numel() > 0 else 0.0)
                    if all(abs(v) <= 1e-12 for v in true_vals):
                        skipped_zero_configs += 1
                        continue
                except Exception:
                    # 若真值读取失败，不在这里静默丢样本，保持原有流程
                    pass

            # --- 关键修改：获取 AST 原始路径 ---
            # 这里优先获取用于 extract_func 解析的原始路径字符串
            # 兼容不同 entry 结构中的 key 名
            raw_ast_path = entry.get('ast_raw_path') or entry.get('ast_path') or entry.get('ast')

            # 如果 raw_ast_path 还是 None，根据 cdfg 路径进行逻辑推导（保底方案）
            if raw_ast_path is None and 'cdfg' in entry:
                # 假设路径结构类似 .../dataset/kernel/data_x.pt
                raw_ast_path = str(entry['cdfg'])

            metric_ids = [self.target_m_id] if self.target_m_id is not None else range(5)
            for m_id in metric_ids:
                candidate_metric_slots += 1
                # 构造与 labels.json 一致的 Key: (kernel_name, filename, metric_id)
                label_key = (k_name, f"data_{c_idx}.pt", m_id)

                if label_key in self.label_dict:
                    self.samples.append({
                        'info': entry,
                        'm_id': m_id,
                        'label': torch.tensor(self.label_dict[label_key], dtype=torch.float32),
                        'ast_path': raw_ast_path  # 【修复 KeyError】确保存入路径
                    })
                    valid_count += 1
                else:
                    skipped_missing_label += 1

        if self.skip_all_zero_config:
            print(f"[过滤统计] 配置总数: {checked_configs}, 全0配置跳过: {skipped_zero_configs}, 保留配置: {checked_configs - skipped_zero_configs}")
        else:
            print(f"[过滤统计] 配置总数: {checked_configs}, 全0配置过滤: 关闭")
        if self.target_m_id is not None:
            print(f"[单指标模式] target_m_id={self.target_m_id}, target_name={target_names[self.target_m_id]}")
        print(
            f"[标签对齐统计] 候选槽位={candidate_metric_slots}, 命中labels={valid_count}, "
            f"缺失labels跳过={skipped_missing_label}"
        )
        if candidate_metric_slots > 0:
            coverage = 100.0 * valid_count / candidate_metric_slots
            print(f"[标签覆盖率] {coverage:.2f}%")
        print(f"[完成] 匹配成功的样本总量: {len(self.samples)}")
        if len(self.samples) == 0:
            raise ValueError("无法匹配任何标签！请检查 label_key 构造逻辑。")

    def __getitem__(self, idx):
        s = self.samples[idx]
        info = s['info']

        # 1. 加载图数据 (CDFG)
        data_obj = torch.load(info['cdfg'], map_location='cpu')

        # 2. 加载高维特征 (Text/AST Embeddings)
        h_s_raw = torch.load(info['text'], map_location='cpu')
        h_a_raw = torch.load(info['ast'], map_location='cpu')

        # 处理维度：确保转换为 [dim] 向量
        h_s = h_s_raw.x.float() if hasattr(h_s_raw, 'x') else h_s_raw.float()
        h_a = h_a_raw.x.float() if hasattr(h_a_raw, 'x') else h_a_raw.float()

        if h_s.dim() > 1: h_s = h_s.mean(dim=0)
        if h_a.dim() > 1: h_a = h_a.mean(dim=0)

        # 3. 挂载特征到 data_obj (可选，方便某些模型内部调用)
        data_obj.text_feat = h_s
        data_obj.ast_feat = h_a

        # 4. 混合元特征（kernel 聚合时已对各配置 meta 取平均并缓存）
        if self.aggregate_by_kernel and "meta_feat" in s:
            meta_feat = s["meta_feat"]
        else:
            meta_feat = self.extract_func(s["ast_path"])

        kn = s.get("kernel_name")
        if kn is not None:
            data_obj.kernel = kn

        # 返回 6 个值，严格对应 train_ppo_style 中的解包顺序
        # (data, h_s, h_a, meta, m_id, label)
        return data_obj, h_s, h_a, meta_feat, s['m_id'], s['label']

    def __len__(self):
        return len(self.samples)

def load_labels_from_json(path):
    if not os.path.exists(path): return {}
    with open(path, 'r') as f:
        raw = json.load(f)
    # 注意：确保 key 的解析与 Dataset 中构建的 key 一致
    return {eval(k): v for k, v in raw.items()}
