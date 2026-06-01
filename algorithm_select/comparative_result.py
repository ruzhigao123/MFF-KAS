import os
import torch
import torch.nn as nn
import sys
import numpy as np
import builtins
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

# --- 1. 路径与环境配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(BASE_DIR)

from src.model import Net
from src.utils import MLP as MLP_Class, _get_y_with_target
from make_labels import get_aligned_data, MockFlags, get_feat

# 测试集白名单 (Unseen Kernels)
MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'doitgen',
               'mvt', 'fdtd-2d', 'gemver', 'gemm-p', 'gesummv',
               'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d']


class ComparativeEvaluator:
    def __init__(self, weight_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not os.path.isabs(weight_dir):
            self.weight_dir = os.path.join(CURRENT_DIR, weight_dir)
        else:
            self.weight_dir = weight_dir

        self.target_names = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']

        # 7个专家的信息
        self.tools_info = [
            {"id": 0, "file": "all_regression_model_state_dict.pth", "mode": "all", "dim": 153},
            {"id": 1, "file": "ast_and_cdfg_regression_model_state_dict.pth", "mode": "ast_cdfg", "dim": 153},
            {"id": 2, "file": "ast_and_text_regression_model_state_dict.pth", "mode": "ast_text", "dim": 153},
            {"id": 3, "file": "cdfg_and_text_regression_model_state_dict.pth", "mode": "cdfg_text", "dim": 153},
            {"id": 4, "file": "cdfg_regression_model_state_dict.pth", "mode": "cdfg_only", "dim": 153},
            {"id": 5, "file": "ast_regression_model_state_dict.pth", "mode": "ast_only", "dim": 784},
            {"id": 6, "file": "text_regression_model_state_dict.pth", "mode": "LM", "dim": 153}
        ]

    def _get_test_data(self):
        all_aligned = get_aligned_data()[0]
        seen_mach = set(MACHSUITE_KERNEL[:5])
        seen_poly = set(poly_KERNEL[:10])
        test_list = []
        for info in all_aligned:
            domain, kname, _ = info['key']
            is_test = False
            if "mach" in domain.lower():
                if kname not in seen_mach: is_test = True
            elif "poly" in domain.lower():
                if kname not in seen_poly: is_test = True
            if is_test: test_list.append(info)
        return test_list

    def run_evaluation(self):
        test_list = self._get_test_data()
        print(f"📊 权重目录: {self.weight_dir}")
        print(f"📊 测试样本数: {len(test_list)}")

        results_report = {}

        for tool in self.tools_info:
            mode = tool['mode']
            ckpt_path = os.path.join(self.weight_dir, tool['file'])
            if not os.path.exists(ckpt_path):
                print(f"⚠️ 缺失权重: {mode}")
                continue

            print(f">>> 评估专家 #{tool['id']}: {mode}")

            # 初始化模型
            builtins.FLAGS = MockFlags(mode)
            model = Net(in_channels=tool['dim'], target_list=self.target_names).to(self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device)

            # --- 关键：针对 LM 模式加载或初始化 MLP1 ---
            if mode == "LM":
                # 打印所有键以供调试（如果再次失败，请查看控制台输出的 Keys）
                # print(f"DEBUG: ckpt keys = {ckpt.keys()}")

                # 尝试寻找降维层（可能是 MLP1, 也可能是其他名字）
                potential_mlp_keys = [k for k in ckpt.keys() if "MLP1" in k and "weight" in k]

                if potential_mlp_keys:
                    try:
                        # 自动获取第一个匹配到的权重键
                        target_key = potential_mlp_keys[0]
                        out_dim, in_dim = ckpt[target_key].shape

                        # 这里的 MLP_Class 必须与你训练时的结构一致
                        # 如果训练时 MLP1 只有一层，这里 layers 应设为 1
                        model.MLP1 = MLP_Class(in_dim, out_dim, 'relu', 2, [128, 128]).to(self.device)
                        print(f"   [OK] 已成功从键 {target_key} 初始化 MLP1 ({in_dim} -> {out_dim})")
                    except Exception as e:
                        print(f"   [Error] 自动初始化 MLP1 失败: {e}")
                else:
                    print(f"   [Warning] 权重中未发现 MLP1 相关键，将尝试直接映射。")

            # 动态调整输出层维度
            head_key = f"MLPs.{self.target_names[0]}.layers.0.weight"
            exp_dim = ckpt[head_key].shape[1] if head_key in ckpt else 128
            if exp_dim != model.MLPs[self.target_names[0]].layers[0].in_features:
                for name in self.target_names:
                    model.MLPs[name] = MLP_Class(exp_dim, 1, 'relu', 2, [128, 128]).to(self.device)

            model.load_state_dict(ckpt, strict=False)
            model.eval()

            storage = defaultdict(lambda: {t: {'true': [], 'pred': []} for t in self.target_names})

            with torch.no_grad():
                for info in tqdm(test_list, desc=f"Infer {mode}", leave=False):
                    k_name = info['key'][1]
                    d_cdfg = torch.load(info['cdfg'], map_location='cpu')
                    d_text = torch.load(info['text'], map_location='cpu')
                    d_ast = torch.load(info['ast'], map_location='cpu')

                    t_f = get_feat(d_text).to(self.device)
                    h_s_raw = global_mean_pool(t_f, torch.zeros(t_f.size(0), dtype=torch.long, device=self.device))

                    # --- 针对 LM 专家的特殊处理：动态创建 MLP1 ---
                    if mode == "LM":
                        # 打印所有键以供调试（如果再次失败，请查看控制台输出的 Keys）
                        # print(f"DEBUG: ckpt keys = {ckpt.keys()}")

                        # 尝试寻找降维层（可能是 MLP1, 也可能是其他名字）
                        potential_mlp_keys = [k for k in ckpt.keys() if "MLP1" in k and "weight" in k]

                        if potential_mlp_keys:
                            try:
                                # 自动获取第一个匹配到的权重键
                                target_key = potential_mlp_keys[0]
                                out_dim, in_dim = ckpt[target_key].shape

                                # 这里的 MLP_Class 必须与你训练时的结构一致
                                # 如果训练时 MLP1 只有一层，这里 layers 应设为 1
                                model.MLP1 = MLP_Class(in_dim, out_dim, 'relu', 2, [128, 128]).to(self.device)
                                print(f"   [OK] 已成功从键 {target_key} 初始化 MLP1 ({in_dim} -> {out_dim})")
                            except Exception as e:
                                print(f"   [Error] 自动初始化 MLP1 失败: {e}")
                        else:
                            print(f"   [Warning] 权重中未发现 MLP1 相关键，将尝试直接映射。")

                    elif mode in ['ast_only', 'cdfg_only']:
                        current_graph = d_ast if mode == 'ast_only' else d_cdfg
                        batch_graph = Batch.from_data_list([current_graph.to(self.device)])
                        input_to_mlp = model.gnn_model(batch_graph)
                    else:
                        # 融合模型
                        batch_graph = Batch.from_data_list([d_cdfg.to(self.device)])
                        h_g = model.gnn_model(batch_graph)
                        a_f = get_feat(d_ast).to(self.device)
                        h_a_raw = global_mean_pool(a_f, torch.zeros(a_f.size(0), dtype=torch.long, device=self.device))
                        h_s, h_a = h_s_raw, h_a_raw
                        if mode == 'ast_text':
                            h_g = torch.zeros_like(h_g)
                        elif mode == 'ast_cdfg':
                            h_s = torch.zeros_like(h_s)
                        elif mode == 'cdfg_text':
                            h_a = torch.zeros_like(h_a)
                        input_to_mlp = model.fusion_block(h_g=h_g, h_s=h_s, h_a=h_a)

                    for m_name in self.target_names:
                        true_val = _get_y_with_target(d_cdfg, m_name).item()
                        pred_val = model.MLPs[m_name](input_to_mlp).cpu().item()
                        storage[k_name][m_name]['true'].append(true_val)
                        storage[k_name][m_name]['pred'].append(pred_val)

            results_report[mode] = storage
            del model
            torch.cuda.empty_cache()

        self._print_final_table(results_report)

    def _print_final_table(self, results_report):
        """生成要求的横向对比大表"""
        modes = [t['mode'] for t in self.tools_info]
        valid_modes = [m for m in modes if results_report.get(m) is not None]
        if not valid_modes: return

        all_kernels = sorted(results_report[valid_modes[0]].keys())

        # 打印表头
        header = f"{'Kernel':<15} | {'Target':<10} |"
        for m in modes:
            header += f" {m[:10]:^11} |"
        print("\n" + "=" * 125)
        print(header)
        print("-" * 125)

        expert_total_mse = {m: [] for m in modes}

        for kn in all_kernels:
            for target in self.target_names:
                row = f"{kn[:15]:<15} | {target:<10} |"
                for m in modes:
                    if m in results_report and kn in results_report[m]:
                        t_data = results_report[m][kn][target]
                        mse = np.mean((np.array(t_data['true']) - np.array(t_data['pred'])) ** 2)
                        row += f" {mse:.4e} |"
                        expert_total_mse[m].append(mse)
                    else:
                        row += f" {'N/A':^11} |"
                print(row)
            print("-" * 125)

        # 打印 OVERALL
        print("\n" + "=" * 30 + " 专家维度汇总 (7 Kernels 平均) " + "=" * 30)

        for m in modes:
            if m not in results_report or results_report[m] is None:
                continue

            print(f"\n>> 专家: {m.upper()}")
            print(f"{'Metric':<12} | {'Avg MSE':<15} | {'Avg RMSE':<15}")
            print("-" * 46)

            mode_metrics_mse = []
            storage = results_report[m]
            kernels = list(storage.keys())

            for target in self.target_names:
                # 收集该专家在当前指标下，所有 Kernel 的 MSE
                target_mses = []
                for kn in kernels:
                    true_val = np.array(storage[kn][target]['true'])
                    pred_val = np.array(storage[kn][target]['pred'])
                    mse = np.mean((true_val - pred_val) ** 2)
                    target_mses.append(mse)

                avg_mse = np.mean(target_mses)
                avg_rmse = np.sqrt(avg_mse)
                mode_metrics_mse.append(avg_mse)

                print(f"{target:<12} | {avg_mse:.6e} | {avg_rmse:.6e}")

            # 计算该专家的综合平均值
            overall_mse = np.mean(mode_metrics_mse)
            print("-" * 46)
            print(f"{'OVERALL':<12} | {overall_mse:.6e} | {np.sqrt(overall_mse):.6e}")

        print("=" * 90)


if __name__ == "__main__":
    evaluator = ComparativeEvaluator("../model_select_weight")
    evaluator.run_evaluation()