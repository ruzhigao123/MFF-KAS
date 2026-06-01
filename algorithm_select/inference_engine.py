
import os
import re
import sys
import glob
import builtins
from collections import OrderedDict, Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

# --- 1. 路径与环境配置 ---
# 自动定位工程根目录：.../MPM-Innovation-master/MPM-Innovation-master
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from select_model import AlgorithmSelector  # noqa: E402
from src.model import Net  # noqa: E402
from src.utils import MLP as MLP_Class, _get_y_with_target  # noqa: E402
from src.config import FLAGS as SRC_FLAGS  # noqa: E402
from make_labels import get_aligned_data, MockFlags, get_feat  # noqa: E402
from select_main import extract_hybrid_meta  # noqa: E402


def load_labels_from_json(path: str):
    """
    labels.json: key 形如 "('nw', 'data_342.pt', 0)" -> value 为长度7的误差向量(list)
    """
    import json
    if not os.path.exists(path):
        raise FileNotFoundError(f"labels.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        try:
            out[eval(k)] = v
        except Exception:
            continue
    return out


def _scalar_true_y(d_cdfg, target_name: str) -> float:
    """图节点上的真值取 mean，与训练里对 batch 标量化的方式一致。"""
    y = _get_y_with_target(d_cdfg, target_name).view(-1)
    return float(y.mean().item()) if y.numel() > 0 else 0.0


# 与 data/regression_model_pred_{train|test}_{tag}.csv 及 build_labels_from_data_csv 一致
EXPERT_ID_TO_DATA_CSV_TAG = {
    0: "all",
    1: "ast_and_cdfg",
    2: "ast_and_text",
    3: "cdfg_and_text",
    4: "cdfg",
    5: "ast",
    6: "text",
}

TARGET_NAME_TO_CSV_COLS = {
    "perf": ("perf_pred", "perf_true"),
    "util-LUT": ("util-LUT_pred", "util-LUT_true"),
    "util-FF": ("util-FF_pred", "util-FF_true"),
    "util-DSP": ("util-DSP_pred", "util-DSP_true"),
    "util-BRAM": ("util-BRAM_pred", "util-BRAM_true"),
}


def _normalize_csv_data_filename(file_cell) -> str:
    """与 train 导出 CSV 一致：N.ast.pt -> data_N.pt。"""
    s = str(file_cell).strip()
    base = os.path.basename(s)
    m = re.match(r"^data_(\d+)\.pt$", base, re.I)
    if m:
        return f"data_{int(m.group(1))}.pt"
    m = re.match(r"^(\d+)\.ast\.pt$", base, re.I)
    if m:
        return f"data_{int(m.group(1))}.pt"
    m = re.search(r"(\d+)", base)
    if m and base.endswith(".pt"):
        return f"data_{int(m.group(1))}.pt"
    return base


class RegressionPredCsvStore:
    """
    从项目 data/ 下 regression_model_pred_train_{专家名}.csv / test_{专家名}.csv 建立索引，
    供 RMSE 使用与导出一致的 pred/true（无需再跑专家 Net 前向）。
    专家名：all, ast_and_cdfg, ast_and_text, cdfg_and_text, cdfg, ast, text。
    """

    def __init__(self, data_dir: str):
        self.data_dir = os.path.abspath(data_dir)
        self._cell = {}
        self._file_count = 0
        self._load_all()

    def _tag_to_expert_id(self, tag: str):
        tag = tag.lower()
        for eid, name in EXPERT_ID_TO_DATA_CSV_TAG.items():
            if name == tag:
                return eid
        return None

    def _load_all(self):
        pattern = os.path.join(self.data_dir, "regression_model_pred_*.csv")
        all_fps = sorted(glob.glob(pattern))
        train_f = [p for p in all_fps if "_train_" in os.path.basename(p)]
        test_f = [p for p in all_fps if "_test_" in os.path.basename(p)]
        other_f = [p for p in all_fps if p not in train_f and p not in test_f]
        # 先 train 后 test：与 inference kernel 样本多在 test CSV 一致，同键时 test 覆盖 train
        ordered = train_f + test_f + other_f
        for fp in ordered:
            base = os.path.basename(fp)
            m = re.match(r"^regression_model_pred_(train|test)_(.+)\.csv$", base, re.I)
            if not m:
                continue
            tag = m.group(2).lower()
            if re.fullmatch(r"\d{8}_\d{6}", tag):
                continue
            eid = self._tag_to_expert_id(tag)
            if eid is None:
                continue
            try:
                df = pd.read_csv(fp)
            except Exception:
                continue
            self._file_count += 1
            for _, row in df.iterrows():
                k = str(row.get("kernel", "")).strip()
                fn = _normalize_csv_data_filename(row.get("file", ""))
                if not k or not fn.startswith("data_"):
                    continue
                key = (eid, k, fn)
                bucket = self._cell.setdefault(key, {})
                for tname, (pc, tc) in TARGET_NAME_TO_CSV_COLS.items():
                    if pc not in df.columns or tc not in df.columns:
                        continue
                    try:
                        pv = float(row[pc])
                        tv = float(row[tc])
                    except (TypeError, ValueError):
                        continue
                    bucket[tname] = (pv, tv)

        n = len(self._cell)
        print(
            f">>> RegressionPredCsvStore: dir={self.data_dir}, loaded_csv_files={self._file_count}, "
            f"unique (expert,kernel,file) keys={n}"
        )

    def get(self, expert_id: int, kernel: str, data_basename: str, target_name: str):
        fn = _normalize_csv_data_filename(data_basename)
        key = (int(expert_id), str(kernel).strip(), fn)
        b = self._cell.get(key)
        if not b:
            return None
        return b.get(target_name)


# 定义划分
MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'doitgen',
               'mvt', 'fdtd-2d', 'gemver', 'gemm-p', 'gesummv',
               'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d']


def _kernel_meta_feat_25(valid_entries):
    """
    与 select_dataset.prepare_kernel_aggregated_samples 一致：各配置 extract_hybrid_meta 取前 9 维再平均，后 16 维填 0。
    valid_entries: 同一 kernel 下已通过全零过滤的 info 列表。
    返回 [1, 25] float32 CPU。
    """
    rows = []
    for e in sorted(valid_entries, key=lambda x: x["key"][2]):
        ast_json = str(e["ast"]).replace(".ast.pt", ".ast.json")
        m = extract_hybrid_meta(ast_json)
        if not isinstance(m, torch.Tensor):
            m = torch.as_tensor(m, dtype=torch.float32)
        v = m.float().flatten()
        if v.numel() >= 9:
            rows.append(v[:9].cpu())
        else:
            pad = torch.zeros(9, dtype=torch.float32)
            pad[: v.numel()] = v.cpu().float()
            rows.append(pad)
    meta9 = torch.stack(rows, dim=0).mean(dim=0)
    meta25 = torch.cat([meta9, torch.zeros(16, dtype=torch.float32)], dim=0).unsqueeze(0)
    return meta25


def _pick_representative_entry(valid_entries):
    """配置编号排序后取中位 c_idx 对应 entry（与 Dataset 一致）。"""
    by_c = {}
    for e in valid_entries:
        by_c[e["key"][2]] = e
    c_sorted = sorted(by_c.keys())
    if not c_sorted:
        return None
    repr_c = c_sorted[len(c_sorted) // 2]
    return by_c[repr_c]


class InferenceEngine:
    """
    推理引擎（与最新 select_main.py 对齐）：
    - selector 权重支持按指标分别保存：best_ppo_model_{perf|lut|ff|dsp|bram}.pth
    - 若对应指标权重不存在，则回退到 best_ppo_model_all.pth 或传入的 model_path。
    - selector_ckpt_suffix: 与 select_main_data_is_random 一致，kernel 级样本训练时
      保存为 best_ppo_model_{metric}_kernel.pth，此处传 \"_kernel\"；配置级传 \"\" 。
    """

    METRIC_NAMES = ["perf", "lut", "ff", "dsp", "bram"]

    def __init__(self, model_path: str, weight_dir: str, selector_dir: str = None, selector_ckpt_suffix: str = "_kernel"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_dir = weight_dir
        self.target_names = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']
        self.selector_dir = selector_dir or os.path.join(BASE_DIR, "save_models_and_data")
        self.fallback_model_path = model_path
        # 例如 \"_kernel\" → best_ppo_model_ff_kernel.pth；\"\" → best_ppo_model_ff.pth
        self.selector_ckpt_suffix = selector_ckpt_suffix or ""

        # selector 缓存：每个 metric_id 一个 selector（lazy load）
        self._selectors = {}

        print(f">>> Selector weights dir: {self.selector_dir}")
        print(f">>> Expert weights dir  : {self.weight_dir}")

        # 懒加载专家回归模型（避免同一 tool_id 在评估循环中重复 torch.load）
        self._expert_model_cache = {}

        # 专家模型配置映射（跑专家回归 pred vs 真值 RMSE 时使用）
        self.tools_info = [
            {"id": 0, "file": "all_regression_model_state_dict.pth", "mode": "all", "dim": 153},
            {"id": 1, "file": "ast_and_cdfg_regression_model_state_dict.pth", "mode": "ast_cdfg", "dim": 153},
            {"id": 2, "file": "ast_and_text_regression_model_state_dict.pth", "mode": "ast_text", "dim": 153},
            {"id": 3, "file": "cdfg_and_text_regression_model_state_dict.pth", "mode": "cdfg_text", "dim": 153},
            {"id": 4, "file": "cdfg_regression_model_state_dict.pth", "mode": "cdfg_only", "dim": 153},
            {"id": 5, "file": "ast_regression_model_state_dict.pth", "mode": "ast_only", "dim": 784},
            {"id": 6, "file": "text_regression_model_state_dict.pth", "mode": "LM", "dim": 153},
        ]

    def _resolve_selector_path(self, metric_id: int) -> str:
        """
        与 select_main_data_is_random.py 的保存命名一致：
        - best_ppo_model_{metric}{suffix}.pth（suffix 如 _kernel）
        - best_ppo_model_{metric}.pth（suffix 非空时作为回退）
        - best_ppo_model_all{suffix}.pth / best_ppo_model_all.pth
        """
        metric_name = None
        if metric_id is not None and 0 <= int(metric_id) < len(self.METRIC_NAMES):
            metric_name = self.METRIC_NAMES[int(metric_id)]

        suf = self.selector_ckpt_suffix or ""

        candidates = []
        if metric_name is not None:
            candidates.append(os.path.join(self.selector_dir, f"best_ppo_model_{metric_name}{suf}.pth"))
            if suf:
                candidates.append(os.path.join(self.selector_dir, f"best_ppo_model_{metric_name}.pth"))
        candidates.append(os.path.join(self.selector_dir, f"best_ppo_model_all{suf}.pth"))
        if suf:
            candidates.append(os.path.join(self.selector_dir, "best_ppo_model_all.pth"))
        if self.fallback_model_path:
            candidates.append(self.fallback_model_path)

        for p in candidates:
            if p and os.path.exists(p):
                return p

        # 没找到则返回空，让上层决定如何处理（跳过或报错）
        return ""

    def _get_selector(self, metric_id: int) -> AlgorithmSelector:
        mid = int(metric_id)
        if mid in self._selectors:
            return self._selectors[mid]

        model_path = self._resolve_selector_path(mid)
        if not model_path or not os.path.exists(model_path):
            metric_name = self.METRIC_NAMES[mid] if 0 <= mid < len(self.METRIC_NAMES) else str(mid)
            suf = self.selector_ckpt_suffix or ""
            tried = [
                os.path.join(self.selector_dir, f"best_ppo_model_{metric_name}{suf}.pth"),
                os.path.join(self.selector_dir, f"best_ppo_model_all{suf}.pth"),
            ]
            if suf:
                tried.extend(
                    [
                        os.path.join(self.selector_dir, f"best_ppo_model_{metric_name}.pth"),
                        os.path.join(self.selector_dir, "best_ppo_model_all.pth"),
                    ]
                )
            if self.fallback_model_path:
                tried.append(self.fallback_model_path)
            tried = [t for t in tried if t]
            raise FileNotFoundError(
                f"Selector weight not found for metric_id={mid}. Tried: {tried}"
            )

        selector = AlgorithmSelector(
            gcn_params={'hidden_channels': 128},
            num_tools=7
        ).to(self.device)
        selector.load_state_dict(torch.load(model_path, map_location=self.device))
        selector.eval()

        self._selectors[mid] = selector
        print(f">>> Loaded selector for metric_id={mid} from: {model_path}")
        return selector

    def recommend_tool_argmax(self, raw_cdfg, raw_text, raw_ast, metric_id: int, meta_feat_override=None) -> int:
        """
        与 kernel 级训练/验证一致：在代表图 + meta 上跑一次 selector，取 logits argmax 作为该 kernel 的固定专家。
        无优先级规则、无 TopK 中位数。
        """
        pyg_data = raw_cdfg.to(self.device)
        m_id_tensor = torch.tensor([metric_id], dtype=torch.long).to(self.device)

        t_f_raw = get_feat(raw_text).to(self.device)
        h_s_input = global_mean_pool(t_f_raw, torch.zeros(t_f_raw.size(0), dtype=torch.long, device=self.device))

        a_f_raw = get_feat(raw_ast).to(self.device)
        h_a_input = global_mean_pool(a_f_raw, torch.zeros(a_f_raw.size(0), dtype=torch.long, device=self.device))

        selector = self._get_selector(metric_id)
        with torch.no_grad():
            if meta_feat_override is not None:
                meta_feat = meta_feat_override.to(self.device)
                if meta_feat.dim() == 1:
                    meta_feat = meta_feat.unsqueeze(0)
            else:
                json_path = pyg_data.ast_path[0] if isinstance(pyg_data.ast_path, list) else pyg_data.ast_path
                meta_feat = extract_hybrid_meta(json_path).unsqueeze(0).to(self.device)
            logits = selector(pyg_data, h_s_input, h_a_input, meta_feat, m_id_tensor)
            tool_id = int(logits.argmax(dim=1).item())
        return tool_id

    def recommend_topk_tools(self, raw_cdfg, raw_text, raw_ast, metric_id: int, k: int = 3, meta_feat_override=None):
        """
        只做 selector 排名，返回 Top-K 工具ID（不做 priority 规则重写）。
        meta_feat_override: 可选 [1,25] 或 [25]，与 kernel 聚合训练一致时传入。
        """
        pyg_data = raw_cdfg.to(self.device)
        m_id_tensor = torch.tensor([metric_id], dtype=torch.long).to(self.device)

        t_f_raw = get_feat(raw_text).to(self.device)
        h_s_input = global_mean_pool(t_f_raw, torch.zeros(t_f_raw.size(0), dtype=torch.long, device=self.device))

        a_f_raw = get_feat(raw_ast).to(self.device)
        h_a_input = global_mean_pool(a_f_raw, torch.zeros(a_f_raw.size(0), dtype=torch.long, device=self.device))

        selector = self._get_selector(metric_id)
        with torch.no_grad():
            if meta_feat_override is not None:
                meta_feat = meta_feat_override.to(self.device)
                if meta_feat.dim() == 1:
                    meta_feat = meta_feat.unsqueeze(0)
            else:
                json_path = pyg_data.ast_path[0] if isinstance(pyg_data.ast_path, list) else pyg_data.ast_path
                meta_feat = extract_hybrid_meta(json_path).unsqueeze(0).to(self.device)
            logits = selector(pyg_data, h_s_input, h_a_input, meta_feat, m_id_tensor)
            k_eff = min(k, logits.shape[1])
            _, topk_indices = torch.topk(logits, k_eff, dim=1)
            topk_ids = topk_indices.squeeze().tolist()
            if isinstance(topk_ids, int):
                topk_ids = [topk_ids]
        return [int(x) for x in topk_ids]

    def _ensure_expert_cached(self, tool_id: int):
        """按 tool_id 懒加载并缓存专家 Net（每个 tool 对应不同 MockFlags / 结构）。"""
        tool_id = int(tool_id)
        if tool_id in self._expert_model_cache:
            return self._expert_model_cache[tool_id]
        t_cfg = self.tools_info[tool_id]
        mode = t_cfg["mode"]
        builtins.FLAGS = MockFlags(mode)
        expert_in_channels = 784 if mode == "ast_only" else 153
        expert_model = Net(in_channels=expert_in_channels, target_list=self.target_names).to(self.device)
        ckpt_path = os.path.join(self.weight_dir, t_cfg["file"])
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Expert checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        first_mlp_key = f"MLPs.{self.target_names[0]}.layers.0.weight"
        weight_in_dim = 256
        if first_mlp_key in ckpt:
            weight_in_dim = ckpt[first_mlp_key].shape[1]
            for name in self.target_names:
                expert_model.MLPs[name] = MLP_Class(weight_in_dim, 1, "relu", 2, [128, 128]).to(self.device)
        expert_model.load_state_dict(ckpt, strict=False)
        expert_model.eval()
        self._expert_model_cache[tool_id] = (expert_model, weight_in_dim, mode)
        return self._expert_model_cache[tool_id]

    def expert_regression_predict(
        self, raw_cdfg, raw_text, raw_ast, metric_id, tool_id, h_s_input=None, h_a_input=None
    ):
        """
        指定专家 tool_id，对当前样本做回归预测（与 predict 中专家分支一致）。
        返回标量 pred；失败返回 None。
        """
        tool_id = int(tool_id)
        if h_s_input is None:
            t_f_raw = get_feat(raw_text).to(self.device)
            h_s_input = global_mean_pool(
                t_f_raw, torch.zeros(t_f_raw.size(0), dtype=torch.long, device=self.device)
            )
        if h_a_input is None:
            a_f_raw = get_feat(raw_ast).to(self.device)
            h_a_input = global_mean_pool(
                a_f_raw, torch.zeros(a_f_raw.size(0), dtype=torch.long, device=self.device)
            )
        try:
            expert_model, weight_in_dim, mode = self._ensure_expert_cached(tool_id)
        except FileNotFoundError:
            return None
        builtins.FLAGS = MockFlags(mode)
        with torch.no_grad():
            if mode != "ast_only":
                batch_cdfg = Batch.from_data_list([raw_cdfg.to(self.device)])
                h_g_raw = expert_model.gnn_model(batch_cdfg)
            else:
                batch_ast = Batch.from_data_list([raw_ast.to(self.device)])
                h_g_raw = expert_model.gnn_model(batch_ast)

            if mode == "cdfg_only" or mode == "ast_only":
                input_to_mlp = h_g_raw
            elif mode == "LM":
                input_to_mlp = h_s_input
            else:
                h_g = h_g_raw if ("cdfg" in mode or mode == "all") else torch.zeros_like(h_g_raw)
                h_s = h_s_input if ("text" in mode or mode == "all") else torch.zeros_like(h_s_input)
                h_a = h_a_input if ("ast" in mode or mode == "all") else torch.zeros_like(h_a_input)
                input_to_mlp = expert_model.fusion_block(h_g=h_g, h_s=h_s, h_a=h_a)

            actual_dim = input_to_mlp.shape[1]
            if actual_dim != weight_in_dim:
                if actual_dim > weight_in_dim:
                    input_to_mlp = input_to_mlp[:, :weight_in_dim]
                else:
                    padding = torch.zeros((input_to_mlp.shape[0], weight_in_dim - actual_dim)).to(self.device)
                    input_to_mlp = torch.cat([input_to_mlp, padding], dim=1)

            pred_val = expert_model.MLPs[self.target_names[int(metric_id)]](input_to_mlp).cpu().item()
        return float(pred_val)

    def predict(self, raw_cdfg, raw_text, raw_ast, metric_id, k=3, tool_only=True, meta_feat_override=None):
        """
        与 kernel 级训练对齐的选专家策略（摒弃原优先级 + TopK 中位数）：
        1. 三模态特征经 selector 得到 logits；
        2. 仅取 argmax 作为最终专家 ID。

        k: 仅为兼容旧调用，不参与选专家。
        tool_only=True: 返回 (0.0, tool_id)，不加载专家回归。
        tool_only=False: 用该 tool_id 跑一次 expert_regression_predict 并返回 (pred, tool_id)。

        meta_feat_override: 可选 [1,25]，与 kernel 聚合训练一致；为 None 时从 ast_path 用 extract_hybrid_meta。
        """
        tool_id = self.recommend_tool_argmax(
            raw_cdfg, raw_text, raw_ast, metric_id, meta_feat_override=meta_feat_override
        )
        if tool_only:
            return 0.0, int(tool_id)

        t_f_raw = get_feat(raw_text).to(self.device)
        h_s_input = global_mean_pool(t_f_raw, torch.zeros(t_f_raw.size(0), dtype=torch.long, device=self.device))
        a_f_raw = get_feat(raw_ast).to(self.device)
        h_a_input = global_mean_pool(a_f_raw, torch.zeros(a_f_raw.size(0), dtype=torch.long, device=self.device))
        pred_val = self.expert_regression_predict(
            raw_cdfg, raw_text, raw_ast, metric_id, int(tool_id), h_s_input=h_s_input, h_a_input=h_a_input
        )
        if pred_val is None:
            return 0.0, int(tool_id)
        return float(pred_val), int(tool_id)
def _report_overall(points_dict):
    print("\n" + "=" * 26 + " 总体: RMSE / MAE / MSE (pred vs true) " + "=" * 26)
    print(f"{'Target':<15} | {'RMSE':<15} | {'MAE':<15} | {'MSE':<15}")
    print("-" * 65)
    for target, data in points_dict.items():
        true = np.array(data['true'])
        pred = np.array(data['pred'])
        if len(true) == 0: continue
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        mae = np.mean(np.abs(true - pred))
        mse = np.mean((true - pred) ** 2)
        print(f"{target:<15} | {rmse:.6e} | {mae:.6e} | {mse:.6e}")


def _tool_id_to_kernel_rmse_csv_mode(tool_id: int) -> str:
    """kernel_rmse.csv 表头第二行 mode 与 tools_info['mode'] 对齐（cdfg_only→cdfg，LM→text）。"""
    tid = int(tool_id)
    return {
        0: "all",
        1: "ast_cdfg",
        2: "ast_text",
        3: "cdfg_text",
        4: "cdfg",
        5: "ast",
        6: "text",
    }.get(tid, "all")


def _engine_target_to_kernel_rmse_row(target: str) -> str:
    t = (target or "").strip()
    if t == "perf":
        return "perf"
    if t == "util-LUT":
        return "lut"
    if t == "util-FF":
        return "ff"
    if t == "util-DSP":
        return "dsp"
    if t == "util-BRAM":
        return "bram"
    return t.lower()


def _load_kernel_rmse_csv_lookup(csv_path: str):
    """
    解析 algorithm_select/kernel_rmse.csv：第 1 行 kernel 名，第 2 行 mode，第 3 行起 perf/lut/ff/dsp/bram + RMSE。
    返回 dict[(kernel, mode, row)] -> rmse（row 为小写 perf/lut/...）。
    """
    lookup = {}
    if not csv_path or not os.path.isfile(csv_path):
        return lookup
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as fp:
            lines = [ln.strip() for ln in fp if ln.strip()]
    except OSError:
        return lookup
    if len(lines) < 3:
        return lookup
    r0 = lines[0].split(",")
    r1 = lines[1].split(",")
    max_j = min(len(r0), len(r1))
    for li in range(2, len(lines)):
        parts = lines[li].split(",")
        if len(parts) < 2:
            continue
        row_key = parts[0].strip().lower()
        if row_key not in ("perf", "lut", "ff", "dsp", "bram"):
            continue
        for j in range(1, min(len(parts), max_j)):
            k = str(r0[j]).strip()
            mode = str(r1[j]).strip()
            if not k or not mode:
                continue
            try:
                v = float(parts[j])
            except ValueError:
                continue
            lookup[(k, mode, row_key)] = v
    return lookup


def _report_final_rmse_mse_from_kernel_rmse_csv(
    lookup,
    kernel_metric_argmax_tool,
    inference_kernel_names,
    target_names,
    csv_path,
):
    """
    最终 RMSE/MSE：按 kernel_rmse.csv 中「各 inference kernel × 当前 argmax 专家 mode」取表内 RMSE，
    再对各 kernel 算术平均；MSE 取各 kernel 上 RMSE² 的算术平均（表内无样本数，不做全局 pooled MSE）。
    """
    print("\n" + "=" * 18 + " 最终 RMSE / MSE（kernel_rmse.csv，与 argmax 专家 mode 对齐） " + "=" * 18)
    print(f"(源文件: {os.path.abspath(csv_path)})")
    print(f"{'Target':<15} | {'mean_RMSE@csv':<18} | {'mean_MSE@csv':<18} | {'kernels':<8}")
    print("-" * 72)
    if not lookup:
        print("(表为空或未加载，跳过)")
        return
    for target in target_names:
        row = _engine_target_to_kernel_rmse_row(target)
        rmses = []
        used_k = []
        misses = []
        for k in sorted(inference_kernel_names):
            tid = kernel_metric_argmax_tool.get((k, target))
            if tid is None:
                misses.append(k)
                continue
            mode = _tool_id_to_kernel_rmse_csv_mode(tid)
            key = (k, mode, row)
            if key not in lookup:
                misses.append(f"{k}[{mode}]")
                continue
            rmses.append(float(lookup[key]))
            used_k.append(k)
        if not rmses:
            print(f"{target:<15} | {'N/A':<18} | {'N/A':<18} | 0")
            if misses:
                print(f"    (缺表项或未定 tool: {', '.join(misses[:8])}{'...' if len(misses) > 8 else ''})")
            continue
        mean_rmse = float(np.mean(rmses))
        mean_mse = float(np.mean([r * r for r in rmses]))
        print(f"{target:<15} | {mean_rmse:.6e} | {mean_mse:.6e} | {len(used_k)}")
        if misses:
            print(f"    (未计入 kernel: {', '.join(str(x) for x in misses[:6])}{'...' if len(misses) > 6 else ''})")


def _report_overall_error(err_dict, only_targets=None):
    """
    基于 labels.json 的“误差向量”评估：不再需要加载任何专家回归权重。
    err_dict[target] = {'err': [..], 'tool': [..]}
    输出 RMSE/MAE/MSE 均针对误差本身。
    only_targets: 若为非空 set/list，仅打印这些 target（用于单指标模式）。
    """
    title = " 总体误差报表 (labels.json) "
    if only_targets is not None and len(only_targets) == 1:
        title = f" 总体误差报表 — {next(iter(only_targets))} (labels.json) "
    print("\n" + "=" * 28 + title + "=" * 28)
    print(f"{'Target':<15} | {'RMSE(err)':<15} | {'MAE(err)':<15} | {'MSE(err)':<15}")
    print("-" * 65)
    for target, data in err_dict.items():
        if only_targets is not None and target not in only_targets:
            continue
        errs = np.array(data["err"], dtype=np.float64)
        if errs.size == 0:
            continue
        mse = float(np.mean(errs ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(errs)))
        print(f"{target:<15} | {rmse:.6e} | {mae:.6e} | {mse:.6e}")


def _report_topk_rank_lines(rank_err_dict, metric_label=None):
    """
    打印 3 行：
      - All Top1
      - All Top2
      - All Top3
    metric_label: 单指标评估时在标题中标明指标名。
    """
    mid = f" ({metric_label})" if metric_label else ""
    print("\n" + "=" * 25 + f" Top-K Rank Error Lines{mid} " + "=" * 25)
    print(f"{'Mode':<12} | {'RMSE(err)':<15} | {'MAE(err)':<15} | {'MSE(err)':<15} | {'Count':<8}")
    print("-" * 80)
    for r in [1, 2, 3]:
        errs = np.array(rank_err_dict.get(r, []), dtype=np.float64)
        if errs.size == 0:
            print(f"{f'All Top{r}':<12} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15} | 0")
            continue
        mse = float(np.mean(errs ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(errs)))
        print(f"{f'All Top{r}':<12} | {rmse:.6e} | {mae:.6e} | {mse:.6e} | {errs.size}")


def _report_single_metric_topk_rmse_mse(rank_err_dict, metric_name: str):
    """
    单指标：在逐配置上对 (pred-true) 汇总 RMSE/MSE/MAE。
    Top1 = 代表图+logits 第 1 名专家（与 argmax 同序）；Top2/3 为同一次 forward 的第 2、3 名。
    注意：训练日志里的 acc/ratio 来自 labels 误差向量（make_labels：逐点 |pred-true| 再 mean），
    与这里对「图级标量 pred/true」做的 RMSE 不是同一数值，不可直接等同。
    """
    print("\n" + "=" * 10 + f" {metric_name}: Top-1/2/3 — RMSE/MSE/MAE(pred vs true) " + "=" * 10)
    print(f"{'Rank':<10} | {'RMSE':<18} | {'MAE':<18} | {'MSE':<18} | {'n':<6}")
    print("-" * 78)
    for r in [1, 2, 3]:
        errs = np.array(rank_err_dict.get(r, []), dtype=np.float64)
        if errs.size == 0:
            print(f"{'Top' + str(r):<10} | {'N/A':<18} | {'N/A':<18} | {'N/A':<18} | 0")
            continue
        mse = float(np.mean(errs ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(errs)))
        print(f"{'Top' + str(r):<10} | {rmse:.6e} | {mae:.6e} | {mse:.6e} | {errs.size}")


def _report_single_metric_topk_rmse_mse_from_kernel_rmse_csv(
    kernel_rmse_lookup,
    kernel_top3_ids,
    kernel_n_configs,
    metric_name: str,
    inference_kernel_names,
    csv_src_path: str,
):
    """
    用 kernel_rmse.csv 近似「全测试集池化」的 Top1/2/3 RMSE/MSE（无 MAE：表内无 |err|）。

    前提（kernel 聚合、与 pred-vs-true 表同一套设定）：
    - 每个 inference kernel k 上，代表图+logits 得到 Top3 专家 id：kernel_top3_ids[(k, metric)]。
    - 该 kernel 下参与统计的配置数为 n_k：kernel_n_configs[(k, metric)]（与逐配置 pred−true 条数一致）。
    - CSV 格 RMSE_csv(k,e) 表示：在 k 上固定专家 e 时，该指标下的 RMSE（须为 sqrt(mean_i err_i^2) 且与生成 CSV 时的
      样本集合一致）。对 Rank r，在**能查到表项**的 kernel 集合上池化：

        MSE^(r) = ( sum_k n_k * RMSE_csv(k, e_{k,r})^2 ) / ( sum_k n_k )
        RMSE^(r) = sqrt(MSE^(r))

    若某 kernel 缺表项，该 Rank 下该 kernel 不参与分子分母（打印 n_used = 参与求和的配置数）。
    若 CSV 与当前 regression 导出/划分不一致，数值会与「pred vs true」Top 表有偏差；MAE 无法由 RMSE 唯一反推。
    """
    print(
        "\n" + "=" * 8
        + f" {metric_name}: Top-1/2/3 — RMSE/MSE (kernel_rmse.csv, 加权池化 Σ (n_k/N)·RMSE²) "
        + "=" * 8
    )
    print(f"(源文件: {os.path.abspath(csv_src_path)})")
    if not kernel_rmse_lookup:
        print(">>> kernel_rmse 表未加载，跳过。")
        return
    row_key = _engine_target_to_kernel_rmse_row(metric_name)
    ks = sorted(inference_kernel_names)
    N = int(sum(int(kernel_n_configs.get((k, metric_name), 0)) for k in ks))
    if N <= 0:
        print(">>> 无有效 n_k（kernel_n_configs 为空），跳过。")
        return
    print(f"{'Rank':<10} | {'RMSE@csv':<18} | {'MSE@csv':<18} | {'n_used':<8} | {'MAE':<10}")
    print("-" * 78)
    for r in [1, 2, 3]:
        num = 0.0
        den = 0
        miss = []
        for k in ks:
            n_k = int(kernel_n_configs.get((k, metric_name), 0))
            if n_k <= 0:
                continue
            t3 = kernel_top3_ids.get((k, metric_name))
            if not t3 or len(t3) < r:
                miss.append(k)
                continue
            eid = int(t3[r - 1])
            mode = _tool_id_to_kernel_rmse_csv_mode(eid)
            rv = kernel_rmse_lookup.get((k, mode, row_key))
            if rv is None:
                miss.append(f"{k}[{mode}]")
                continue
            fv = float(rv)
            num += float(n_k) * (fv * fv)
            den += n_k
        if miss:
            print(f"  (Rank{r} 缺 Top3 或缺表项: {', '.join(str(x) for x in miss[:6])}{'...' if len(miss) > 6 else ''})")
        if den <= 0:
            print(f"{'Top' + str(r):<10} | {'N/A':<18} | {'N/A':<18} | {0:<8} | {'N/A':<10}")
            continue
        mse_acc = num / float(den)
        rmse_v = float(np.sqrt(mse_acc))
        print(f"{'Top' + str(r):<10} | {rmse_v:.6e} | {mse_acc:.6e} | {den:<8} | {'N/A':<10}")


def _report_topk_rank_lines_by_target(rank_err_by_target):
    """
    按目标分开打印 Top1/Top2/Top3 三行。
    rank_err_by_target[target][rank] = 逐样本 (pred - true) 残差。
    """
    print("\n" + "=" * 16 + " Top-K: RMSE/MSE(pred vs true) By Target " + "=" * 16)
    for target, rank_dict in rank_err_by_target.items():
        print(f"\n[Target: {target}]")
        print(f"{'Mode':<12} | {'RMSE':<15} | {'MAE':<15} | {'MSE':<15} | {'Count':<8}")
        print("-" * 80)
        for r in [1, 2, 3]:
            errs = np.array(rank_dict.get(r, []), dtype=np.float64)
            if errs.size == 0:
                print(f"{f'Top{r}':<12} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15} | 0")
                continue
            mse = float(np.mean(errs ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(errs)))
            print(f"{f'Top{r}':<12} | {rmse:.6e} | {mae:.6e} | {mse:.6e} | {errs.size}")


def _report_per_kernel_sel_err(
    kernel_rmse_lookup,
    kernel_metric_topk_tool,
    metric_display: str,
    kernel_whitelist,
    logits_top_rank: int,
    csv_src_path: str,
    *,
    no_kernel_rmse_table: bool,
):
    """
    按 kernel 表：RMSE / MSE 均来自 kernel_rmse.csv（MSE = 表中 RMSE 的平方），
    列对应 (kernel, logits Top-r 专家 mode, 当前指标行)。
    """
    title = (
        f" Test set by kernel — {metric_display}: RMSE/MSE (kernel_rmse.csv), "
        f"logits Top{logits_top_rank} "
    )
    print("\n" + "=" * 6 + title + "=" * 6)
    col_tid = f"Expert@Top{logits_top_rank}"
    if no_kernel_rmse_table:
        print(">>> 已跳过（指定了 --no_kernel_rmse_table）。")
        return
    if not kernel_rmse_lookup:
        print(
            f">>> kernel_rmse 表未加载或为空，无法填表。路径: {os.path.abspath(csv_src_path) if csv_src_path else '(未设置)'}。"
        )
        return
    print(f"(源文件: {os.path.abspath(csv_src_path)})")
    print(f"{'Kernel':<18} | {col_tid:<16} | {'RMSE@csv':<16} | {'MSE@csv':<16}")
    print("-" * 72)
    row_key = _engine_target_to_kernel_rmse_row(metric_display)
    keys = sorted(kernel_whitelist) if kernel_whitelist is not None else sorted(
        {k for (k, m) in kernel_metric_topk_tool if m == metric_display}
    )
    for k in keys:
        tid = kernel_metric_topk_tool.get((k, metric_display))
        tid_disp = str(int(tid)) if tid is not None else "N/A"
        if tid is None:
            print(f"{k:<18} | {tid_disp:<16} | {'N/A':<16} | {'N/A':<16}")
            continue
        mode = _tool_id_to_kernel_rmse_csv_mode(int(tid))
        key = (k, mode, row_key)
        rmse_v = kernel_rmse_lookup.get(key)
        if rmse_v is None:
            print(f"{k:<18} | {tid_disp:<16} | {'N/A':<16} | {'N/A':<16}  (无表项 {k},{mode},{row_key})")
        else:
            rv = float(rmse_v)
            mse_v = rv * rv
            print(f"{k:<18} | {tid_disp:<16} | {rv:.6e} | {mse_v:.6e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inference: kernel 级与 select_main_data_is_random 对齐——代表图+聚合 meta 上 logits argmax 固定专家；"
        "RMSE 默认从 <项目>/data/regression_model_pred_*_*.csv 读 pred/true（与 train 导出一致），"
        "传 --no_data_csv_rmse 才改为专家 .pth 前向。"
    )
    parser.add_argument(
        "--metric_id",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="If set, only evaluate this metric id (0:perf,1:lut,2:ff,3:dsp,4:bram).",
    )
    parser.add_argument(
        "--skip_all_zero_config",
        type=bool,
        default=bool(getattr(SRC_FLAGS, "skip_all_zero_config", True)),
        help="Skip configs whose true perf/lut/ff/dsp/bram are all 0 (same preprocessing as training).",
    )
    parser.add_argument(
        "--per_config_eval",
        action="store_true",
        help="Force per-configuration evaluation (disable kernel-level aggregation).",
    )
    _kt_default = 2
    _env_kt = os.environ.get("INFER_KERNEL_TABLE_RANK")
    if _env_kt is not None:
        try:
            _v = int(_env_kt)
            if _v in (1, 2, 3):
                _kt_default = _v
        except ValueError:
            pass
    parser.add_argument(
        "--kernel_table_rank",
        type=int,
        default=_kt_default,
        choices=[1, 2, 3],
        help="kernel 聚合下单指标「按 kernel 表」：代表图 logits 第几名 (1..3) 的专家 id，"
        "用于在 kernel_rmse.csv 中查该 kernel×mode 的 RMSE；Overall 主表仍用 argmax 固定专家。"
        "环境变量 INFER_KERNEL_TABLE_RANK 可改默认。",
    )
    parser.add_argument(
        "--selector_no_kernel_suffix",
        action="store_true",
        help="加载 best_ppo_model_{metric}.pth（无 _kernel），与 select_main sample_granularity=config 一致；"
        "默认使用 _kernel 后缀（best_ppo_model_ff_kernel.pth）。",
    )
    parser.add_argument(
        "--use_data_csv_rmse",
        action="store_true",
        help="兼容旧命令：默认已从 CSV 读 pred/true，无需再传本项。",
    )
    parser.add_argument(
        "--no_data_csv_rmse",
        action="store_true",
        help="关闭 CSV：RMSE 改为各专家 .pth 前向 + 图上真值（默认开启从 data 下 regression_model_pred_*.csv 读取）。",
    )
    parser.add_argument(
        "--data_pred_csv_dir",
        type=str,
        default=None,
        help="regression_model_pred_*.csv 所在目录，默认 <项目根>/data。",
    )
    parser.add_argument(
        "--kernel_rmse_csv",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_rmse.csv"),
        help="kernel×专家 mode 的 RMSE 宽表（与 select_main 导出格式一致）；kernel 聚合模式下据此打印最终 mean_RMSE / mean_MSE。",
    )
    parser.add_argument(
        "--no_kernel_rmse_table",
        action="store_true",
        help="禁用基于 kernel_rmse.csv 的最终 RMSE/MSE 汇总。",
    )
    args = parser.parse_args()
    # 默认用 CSV 与 train 导出一致；仅当显式 --no_data_csv_rmse 时才跑专家 Net
    use_data_csv_rmse = not args.no_data_csv_rmse

    # 与 kernel 聚合训练对齐：默认 kernel 级一次 selector + 聚合 meta；INFER_EVAL_KERNEL_AGG=0 或 --per_config_eval 关闭
    EVAL_KERNEL_AGGREGATE = (
        os.environ.get("INFER_EVAL_KERNEL_AGG", "1").lower() not in ("0", "false", "no")
    ) and not args.per_config_eval

    # 指定 --metric_id 时：报表只打该指标 + 各 inference kernel 的 RMSE/MSE，并缩短启动日志
    single_metric_mode = args.metric_id is not None
    _TNAMES = ("perf", "util-LUT", "util-FF", "util-DSP", "util-BRAM")

    # 与 select_main.py 对齐：默认从 save_models_and_data/ 下加载 best_ppo_model_{metric}.pth
    # 若单指标权重不存在，会自动回退到 best_ppo_model_all.pth
    MODEL_PATH = os.path.join(BASE_DIR, "save_models_and_data/best_ppo_model_all.pth")
    EXP_DIR = os.path.join(BASE_DIR, "model_select_weight")
    SEL_DIR = os.path.join(BASE_DIR, "save_models_and_data")

    # 1. 建立数据索引
    all_aligned, _ = get_aligned_data()

    # 与 src/train.py 对齐：Kernel 级别随机划分（seed=64, 15 train kernels + 7 inference kernels）
    all_kernels = MACHSUITE_KERNEL + poly_KERNEL
    num_train_kernels = 15
    np.random.seed(64)
    shuffled_indices = np.random.permutation(len(all_kernels))
    train_kernel_names = set([all_kernels[i] for i in shuffled_indices[:num_train_kernels]])
    inference_kernel_names = set([all_kernels[i] for i in shuffled_indices[num_train_kernels:]])
    if not single_metric_mode:
        print(f">>> Train kernels (15): {sorted(list(train_kernel_names))}")
        print(f">>> Inference kernels (7): {sorted(list(inference_kernel_names))}")
    else:
        print(f">>> Inference kernels ({len(inference_kernel_names)}): {sorted(list(inference_kernel_names))}")
        print(f">>> Single-metric eval: {_TNAMES[args.metric_id]} (metric_id={args.metric_id})")
        print(
            ">>> 主评估: argmax 固定专家；Top1/2/3 表: 代表图(或逐配置)上 logits 名次专家 — pred vs true；"
            f"按 kernel 表: kernel_rmse.csv 中 RMSE（logits Top{args.kernel_table_rank} 对应专家列）。"
        )

    # 2. 筛选测试集 (Inference / Unseen Kernels)
    test_list = []
    for info in all_aligned:
        k_name = info['key'][1]
        if k_name in inference_kernel_names:
            test_list.append(info)

    print(f"🚀 开始测试 Unseen Kernels. 配置级条目数: {len(test_list)} (kernel_agg={EVAL_KERNEL_AGGREGATE})")
    if EVAL_KERNEL_AGGREGATE:
        print(">>> 评估模式: kernel 聚合（代表图 + 聚合 meta；每 kernel×指标 argmax 固定专家，全配置仅用该专家）")
    else:
        print(">>> 评估模式: 逐配置（每配置各自 logits argmax，无优先级/中位数）")

    _sel_suffix = "" if args.selector_no_kernel_suffix else "_kernel"
    engine = InferenceEngine(
        model_path=MODEL_PATH,
        weight_dir=EXP_DIR,
        selector_dir=SEL_DIR,
        selector_ckpt_suffix=_sel_suffix,
    )
    print(f">>> Selector checkpoint suffix: {repr(_sel_suffix)} (best_ppo_model_<metric>{_sel_suffix}.pth)")

    _csv_dir = args.data_pred_csv_dir or os.path.join(BASE_DIR, "data")
    csv_store = RegressionPredCsvStore(_csv_dir) if use_data_csv_rmse else None
    if use_data_csv_rmse:
        print(f">>> RMSE 数据源（默认）: CSV 目录 {os.path.abspath(_csv_dir)}；专家 .pth 前向已跳过。")
    else:
        print(">>> RMSE 数据源: 专家 Net 前向（已指定 --no_data_csv_rmse）。")
    csv_lookup_miss = 0

    _kernel_rmse_path = None if args.no_kernel_rmse_table else (args.kernel_rmse_csv or "").strip()
    _kernel_rmse_lookup = _load_kernel_rmse_csv_lookup(_kernel_rmse_path) if _kernel_rmse_path else {}
    if _kernel_rmse_path and not args.no_kernel_rmse_table:
        ncell = len(_kernel_rmse_lookup)
        print(
            f">>> kernel_rmse 表: {os.path.abspath(_kernel_rmse_path)} "
            f"({'已加载 ' + str(ncell) + ' 个单元' if ncell else '未读到有效单元，最终汇总将跳过'})"
        )

    kernel_metric_argmax_tool = {}
    kernel_metric_topk_tool = {}
    kernel_topk_tid_votes = defaultdict(Counter)
    kernel_top3_ids = {}
    kernel_n_configs = {}

    def _pred_true_rmse(engine_inner, tool_id, k_n, fname, dc, dt, da, m_idx_inner, m_nm):
        """返回 (pred, true) 或 (None, None)。CSV 模式用导出列；否则跑专家回归 + 图真值。"""
        # __main__ 块内嵌套函数无外层 def，csv_lookup_miss 在模块作用域，须用 global 而非 nonlocal
        global csv_lookup_miss
        if csv_store is not None:
            pair = csv_store.get(int(tool_id), k_n, fname, m_nm)
            if pair is None:
                csv_lookup_miss += 1
                return None, None
            return float(pair[0]), float(pair[1])
        pred_v = engine_inner.expert_regression_predict(dc, dt, da, m_idx_inner, tool_id)
        if pred_v is None:
            return None, None
        return float(pred_v), _scalar_true_y(dc, m_nm)
    overall_points = OrderedDict({t: {"true": [], "pred": []} for t in engine.target_names})
    expert_counts = Counter()
    rank_err_by_target = OrderedDict({t: {1: [], 2: [], 3: []} for t in engine.target_names})
    printed_metric_skip = set()
    total_configs = 0
    skipped_all_zero_configs = 0

    def _append_label_stats_kernel(k_name, valid_entries, m_idx, m_name, meta25, d_cdfg, d_text, d_ast):
        """
        - overall_points：代表图 + 聚合 meta 上 logits argmax → fixed_tool_id；该 kernel 下各配置仅用该专家回归（与训练一致）。
        - rank_err Top1/2/3：同一代表图上 recommend_topk_tools 的 logits 第 1/2/3 名专家，在各配置上分别算 pred−true（诊断表）。
        - 「Test set by kernel」RMSE/MSE：kernel_rmse.csv 查表（MSE=RMSE²）；逐配置时用各 kernel 内 TopK 专家 id 众数对齐列名。
        """
        try:
            fixed_tool_id = engine.recommend_tool_argmax(
                d_cdfg, d_text, d_ast, m_idx, meta_feat_override=meta25
            )
            topk_ids = engine.recommend_topk_tools(
                d_cdfg, d_text, d_ast, m_idx, k=3, meta_feat_override=meta25
            )
        except FileNotFoundError as e:
            if m_idx not in printed_metric_skip:
                print(f"[Skip metric_id={m_idx}] {e}")
                printed_metric_skip.add(m_idx)
            return
        expert_counts[int(fixed_tool_id)] += 1
        kernel_metric_argmax_tool[(k_name, m_name)] = int(fixed_tool_id)
        rk_idx_pk = int(args.kernel_table_rank) - 1
        if single_metric_mode and rk_idx_pk < len(topk_ids):
            kernel_metric_topk_tool[(k_name, m_name)] = int(topk_ids[rk_idx_pk])

        n_ok = 0
        for info in valid_entries:
            dc = torch.load(info["cdfg"], map_location="cpu")
            dt = torch.load(info["text"], map_location="cpu")
            da = torch.load(info["ast"], map_location="cpu")
            dc.ast_path = info["ast"].replace(".ast.pt", ".ast.json")
            dc.file = os.path.basename(info["cdfg"])
            fname = os.path.basename(info["cdfg"])
            pred_sel, true_v = _pred_true_rmse(engine, fixed_tool_id, k_name, fname, dc, dt, da, m_idx, m_name)
            if pred_sel is None:
                continue
            overall_points[m_name]["true"].append(true_v)
            overall_points[m_name]["pred"].append(pred_sel)
            n_ok += 1

            for rank_idx, rank in enumerate([1, 2, 3]):
                if rank_idx >= len(topk_ids):
                    continue
                tid = int(topk_ids[rank_idx])
                pred_r, true_r = _pred_true_rmse(engine, tid, k_name, fname, dc, dt, da, m_idx, m_name)
                if pred_r is None:
                    continue
                rank_err_by_target[m_name][rank].append(float(pred_r - true_r))

        if n_ok == 0:
            return
        kernel_top3_ids[(k_name, m_name)] = [int(topk_ids[i]) for i in range(min(3, len(topk_ids)))]
        kernel_n_configs[(k_name, m_name)] = int(n_ok)

    # 3. 推理循环
    if EVAL_KERNEL_AGGREGATE:
        by_k = defaultdict(list)
        for info in test_list:
            by_k[info["key"][1]].append(info)

        for k_name, entries in tqdm(sorted(by_k.items(), key=lambda x: x[0]), desc="Kernel-level eval"):
            valid_entries = []
            for info in entries:
                total_configs += 1
                d_cdfg = torch.load(info["cdfg"], map_location="cpu")
                if args.skip_all_zero_config:
                    try:
                        true_vals = []
                        for t_name in engine.target_names:
                            y = _get_y_with_target(d_cdfg, t_name).view(-1)
                            true_vals.append(float(y.mean().item()) if y.numel() > 0 else 0.0)
                        if all(abs(v) <= 1e-12 for v in true_vals):
                            skipped_all_zero_configs += 1
                            continue
                    except Exception:
                        pass
                valid_entries.append(info)

            if not valid_entries:
                continue
            repr_entry = _pick_representative_entry(valid_entries)
            if repr_entry is None:
                continue
            meta25 = _kernel_meta_feat_25(valid_entries)

            d_cdfg = torch.load(repr_entry["cdfg"], map_location="cpu")
            d_text = torch.load(repr_entry["text"], map_location="cpu")
            d_ast = torch.load(repr_entry["ast"], map_location="cpu")
            d_cdfg.ast_path = repr_entry["ast"].replace(".ast.pt", ".ast.json")
            d_cdfg.file = os.path.basename(repr_entry["cdfg"])

            metric_ids = [args.metric_id] if args.metric_id is not None else list(range(len(engine.target_names)))
            for m_idx in metric_ids:
                m_name = engine.target_names[m_idx]
                _append_label_stats_kernel(k_name, valid_entries, m_idx, m_name, meta25, d_cdfg, d_text, d_ast)
    else:
        for info in tqdm(test_list, desc="Inference Progress"):
            k_name = info['key'][1]
            c_idx = info["key"][2]
            fname = f"data_{c_idx}.pt"
            total_configs += 1

            # 加载三模态数据
            d_cdfg = torch.load(info['cdfg'], map_location='cpu')
            d_text = torch.load(info['text'], map_location='cpu')
            d_ast = torch.load(info['ast'], map_location='cpu')

            # 核心：挂载 ast_path 供特征提取函数使用 (由 .pt 映射到 .json)
            d_cdfg.ast_path = info['ast'].replace('.ast.pt', '.ast.json')
            d_cdfg.file = os.path.basename(info['cdfg'])

            # 训练时的预处理：若该配置 5 个真实指标全为 0，则整配置跳过
            if args.skip_all_zero_config:
                try:
                    true_vals = []
                    for t_name in engine.target_names:
                        y = _get_y_with_target(d_cdfg, t_name).view(-1)
                        true_vals.append(float(y.mean().item()) if y.numel() > 0 else 0.0)
                    if all(abs(v) <= 1e-12 for v in true_vals):
                        skipped_all_zero_configs += 1
                        continue
                except Exception:
                    # 若真值读取失败，不在这里静默丢样本
                    pass

            metric_ids = [args.metric_id] if args.metric_id is not None else list(range(len(engine.target_names)))
            for m_idx in metric_ids:
                m_name = engine.target_names[m_idx]
                # A) argmax 选专家 + TopK（同一 try，避免已写入 overall 后 topk 失败）
                try:
                    _, tool_id = engine.predict(d_cdfg, d_text, d_ast, m_idx, tool_only=True)
                    topk_ids = engine.recommend_topk_tools(d_cdfg, d_text, d_ast, m_idx, k=3)
                except FileNotFoundError as e:
                    if m_idx not in printed_metric_skip:
                        print(f"[Skip metric_id={m_idx}] {e}")
                        printed_metric_skip.add(m_idx)
                    continue
                expert_counts[int(tool_id)] += 1

                fname_pc = os.path.basename(info["cdfg"])
                pred_sel, true_v = _pred_true_rmse(
                    engine, tool_id, k_name, fname_pc, d_cdfg, d_text, d_ast, m_idx, m_name
                )
                if pred_sel is None:
                    continue
                overall_points[m_name]["true"].append(true_v)
                overall_points[m_name]["pred"].append(pred_sel)

                rk_idx = int(args.kernel_table_rank) - 1
                if rk_idx < len(topk_ids):
                    kernel_topk_tid_votes[(k_name, m_name)][int(topk_ids[rk_idx])] += 1

                for rank_idx, rank in enumerate([1, 2, 3]):
                    if rank_idx >= len(topk_ids):
                        continue
                    tid = int(topk_ids[rank_idx])
                    pred_r, true_r = _pred_true_rmse(
                        engine, tid, k_name, fname_pc, d_cdfg, d_text, d_ast, m_idx, m_name
                    )
                    if pred_r is None:
                        continue
                    rank_err_by_target[m_name][rank].append(float(pred_r - true_r))

    # 逐配置模式：用各 kernel 上「logits Top-r 专家 id」的众数，对齐 kernel_rmse.csv 的一列
    if not EVAL_KERNEL_AGGREGATE and kernel_topk_tid_votes:
        for key, ctr in kernel_topk_tid_votes.items():
            if ctr:
                kernel_metric_topk_tool[key] = int(ctr.most_common(1)[0][0])

    # 4. 输出结果报表
    _targets_report = (
        [engine.target_names[args.metric_id]] if single_metric_mode else list(engine.target_names)
    )
    if (
        EVAL_KERNEL_AGGREGATE
        and _kernel_rmse_lookup
        and not args.no_kernel_rmse_table
        and _kernel_rmse_path
    ):
        _report_final_rmse_mse_from_kernel_rmse_csv(
            _kernel_rmse_lookup,
            kernel_metric_argmax_tool,
            inference_kernel_names,
            _targets_report,
            _kernel_rmse_path,
        )
    elif not args.no_kernel_rmse_table and not EVAL_KERNEL_AGGREGATE:
        print(
            "\n>>> kernel_rmse.csv 最终汇总仅在 kernel 聚合评估（默认）下生成；"
            "当前为逐配置模式（--per_config_eval 或 INFER_EVAL_KERNEL_AGG=0），已跳过表驱动最终行。"
        )
    elif not args.no_kernel_rmse_table and _kernel_rmse_path and not _kernel_rmse_lookup:
        print(
            f"\n>>> kernel_rmse.csv 未加载到有效数据（路径 {os.path.abspath(_kernel_rmse_path)}）；"
            "跳过表驱动最终汇总。"
        )

    if single_metric_mode:
        _mname = engine.target_names[args.metric_id]
        _report_single_metric_topk_rmse_mse(rank_err_by_target[_mname], _mname)
        if (
            EVAL_KERNEL_AGGREGATE
            and _kernel_rmse_lookup
            and not args.no_kernel_rmse_table
            and kernel_top3_ids
        ):
            _report_single_metric_topk_rmse_mse_from_kernel_rmse_csv(
                _kernel_rmse_lookup,
                kernel_top3_ids,
                kernel_n_configs,
                _mname,
                inference_kernel_names,
                _kernel_rmse_path or "",
            )
        _report_per_kernel_sel_err(
            _kernel_rmse_lookup,
            kernel_metric_topk_tool,
            _mname,
            inference_kernel_names,
            int(args.kernel_table_rank),
            _kernel_rmse_path or "",
            no_kernel_rmse_table=args.no_kernel_rmse_table,
        )
        # 主链路：kernel 聚合下每 kernel 用 argmax 固定专家，逐配置 pred vs true（与上表 Top1 数值一致）
        _report_overall({_mname: overall_points[_mname]})
        print(
            "\n>>> 指标说明（与 select_main_data_is_random 训练日志对照）:\n"
            "    - 训练 test acc / ratio: 基于 DataLoader 的 error_vectors（labels.json："
            "各专家 |pred-true| 在图上的 mean；kernel 模式下再对配置取 mean 得到 7 维向量）。"
            "acc = argmax 是否等于该向量 argmin；ratio = (min_err+ε)/(argmax_err+ε) 的 batch 平均。\n"
            "    - 本脚本 RMSE: 默认 pred/true 来自 data 下 regression_model_pred_*_*.csv 与当前指标列；"
            "若使用 --no_data_csv_rmse 则为专家 Net 标量 pred 与 _get_y_with_target 图均值 true。"
            "与训练 L1 标签尺度仍不可直接等同。\n"
            "    - 紧随 pred/true 表之后会打印「Top-1/2/3 — RMSE/MSE (kernel_rmse.csv, 加权池化)」："
            "MSE^(r)=(Σ n_k·RMSE_csv²)/(Σ n_k)，RMSE=√MSE（仅 kernel 聚合且表加载成功）；MAE 列 N/A。\n"
            "    - 「Test set by kernel」：RMSE@csv 为表内直接读取，MSE@csv=RMSE²；Expert@TopK 为代表图 logits 名次；"
            "逐配置模式下该专家 id 为各 kernel 内众数。\n"
            "    - 选择器权重路径见上文 \"Loaded selector for metric_id=...\" 一行。"
        )
    else:
        _report_overall(overall_points)
        _report_topk_rank_lines_by_target(rank_err_by_target)

    if not single_metric_mode:
        if args.skip_all_zero_config:
            kept = total_configs - skipped_all_zero_configs
            print(f"\n>>> All-zero config filter: total={total_configs}, removed={skipped_all_zero_configs}, kept={kept}")

        if printed_metric_skip:
            missing = ", ".join([str(i) for i in sorted(printed_metric_skip)])
            print(f"\n>>> Metrics skipped due to missing selector weights: {missing}")

        print("\n>>> 专家推荐频率分布:")
        for tid, count in expert_counts.most_common():
            print(f"专家 #{tid}: 推荐 {count} 次")
    else:
        if printed_metric_skip:
            missing = ", ".join([str(i) for i in sorted(printed_metric_skip)])
            print(f"\n>>> Metrics skipped due to missing selector weights: {missing}")
    if use_data_csv_rmse and csv_lookup_miss > 0:
        print(f"\n>>> CSV RMSE: lookup misses (no row for expert×kernel×file×target): {csv_lookup_miss}")
