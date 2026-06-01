import os
import glob

from numpy.random import permutation
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import FLAGS, TARGETS
import sys
from saver import saver
from utils import MLP, OurTimer, get_save_path, _get_y_with_target, get_root_path
import programl_data
from torch_geometric.utils import degree
from model import Net
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, \
    mean_absolute_percentage_error, classification_report, confusion_matrix
import torch

import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from collections import defaultdict

def custom_collate(batch):
    from torch_geometric.data import Batch
    import os, torch, re
    for i in range(100):
        print("🔥 DEBUG: custom_collate 开始执行!")
    TEXT_ROOT = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_dataset/code"
    AST_ROOT = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_datasets/code/ast_final_dataset_1"

    text_list, ast_list, final_data_list = [], [], []

    # 只在第一个 batch 的第一个样本打印一次，检查路径对不对
    first_check = True

    for data in batch:
        kernel = getattr(data, 'kernel', None)
        # 从 data_1.pt 提取 1
        idx_match = re.search(r'\d+', str(data.file))
        if not idx_match or kernel is None:
            continue

        cdfg_id = int(idx_match.group())

        # 匹配文本路径
        t_path = None
        for sub in ['poly', 'machsuite']:
            potential_path = os.path.join(TEXT_ROOT, sub, kernel, f"{cdfg_id}.pt")
            if os.path.exists(potential_path):
                t_path = potential_path
                break

        # 匹配 AST 路径 (偏移 -1)
        ast_id = cdfg_id - 1
        a_path = os.path.join(AST_ROOT, f"{ast_id}.ast.pt")

        if first_check:
            print(f"\n🔍 [DEBUG] 正在匹配 Kernel: {kernel}, ID: {cdfg_id}")
            print(f"   尝试文本路径: {t_path}")
            print(f"   尝试 AST 路径: {a_path}")
            first_check = False

        if t_path and os.path.exists(a_path):
            try:
                text_list.append(torch.load(t_path, map_location='cpu'))
                ast_list.append(torch.load(a_path, map_location='cpu'))
                final_data_list.append(data)
            except Exception as e:
                print(f"❌ 加载失败: {e}")

    if len(final_data_list) == 0:
        # 如果这里抛出异常，说明 custom_collate 没能对齐任何一个样本
        raise RuntimeError(f"❌ 严重错误: custom_collate 没能对齐任何样本！请检查上面的路径打印。")

    out_batch = Batch.from_data_list(final_data_list)
    out_batch.text_feat = torch.stack(text_list).to(FLAGS.device)
    out_batch.ast_feat = torch.stack(ast_list).to(FLAGS.device)
    return out_batch

def _report_rmse_etc(points_dict, label, print_result=True):
    # saver.log_info(f"\n>>> Report for: {label}")
    if print_result:
        saver.log_info(label)

    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats import kendalltau

    # 结果字典
    results = {}

    for target_name, d in points_dict.items():
        # 直接获取 true 和 pred 列表 (现在它们是单纯的 float 列表)
        y_true = np.array(d['true'])
        y_pred = np.array(d['pred'])

        if len(y_true) == 0:
            continue

        # 计算基础指标
        mse = mean_squared_error(y_true, y_pred) # <--- 保留 MSE
        rmse = np.sqrt(mse)
        # rmse = mean_squared_error(y_true, y_pred, )
        mae = mean_absolute_error(y_true, y_pred)

        # 计算相对误差 MAPE (避免除以 0)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) if np.any(mask) else 0.0

        # 计算最大误差
        max_err = np.max(np.abs(y_true - y_pred))

        # 计算肯德尔相关系数 (Kendall's Tau) 用于衡量排序相关性
        tau, _ = kendalltau(y_true, y_pred)

        # 记录结果
        results[target_name] = {
            'mse': mse,        # <--- 新增记录 MSE
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'tau': tau,
            'max_err': max_err
        }

        if print_result:
            # 在打印字符串中加入 MSE，同样使用科学计数法
            saver.log_info(
                f"Target: {target_name:15s} | MSE: {mse:.4e} | RMSE: {rmse:.4e} | MAE: {mae:.4e} | MAPE: {mape:.4f} | Tau: {tau:.4f}")

    return results

import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, \
    mean_absolute_percentage_error
from scipy.stats import rankdata, kendalltau
from torch.nn import Sequential, Linear, ReLU
from tqdm import tqdm
from os.path import join
from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
from src.causal_data_utils import hamming_distance, create_intervention_pairs
from src.parameter import DesignPoint
from typing import List, Tuple, Optional

out_dim = FLAGS.out_dim

MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'doitgen',
                'mvt', 'fdtd-2d', 'gemver', 'gemm-p', 'gesummv',
                'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d']


def _is_all_zero_config_sample(data, eps: float = 1e-12) -> bool:
    """
    判断单个配置是否在 5 个真实指标上都为 0。
    """
    metric_names = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']
    vals = []
    for name in metric_names:
        try:
            y = _get_y_with_target(data, name).view(-1)
            vals.append(float(y.mean().item()) if y.numel() > 0 else 0.0)
        except Exception:
            vals.append(0.0)
    return all(abs(v) <= eps for v in vals)



#计算分类损失
def report_class_loss(points_dict):
    d = points_dict[FLAGS.target[0]]
    labels = [data for data,_ in d['pred']]
    pred = [data for _,data in d['pred']]
    target_names = ['invalid', 'valid']
    saver.info('classification report')
    saver.log_info(classification_report(labels, pred, target_names=target_names))
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    saver.info(f'Confusion matrix:\n{cm}')

#计算rmse结果


    num_data = None
    try:
        for target_name, d in points_dict.items():
            # true_li = d['true']
            # pred_li = d['pred']
            true_li = [data for data,_ in d['pred']]
            pred_li = [data for _,data in d['pred']]
            num_data = len(true_li)
            mape = mean_absolute_percentage_error(true_li, pred_li)
            rmse = mean_squared_error(true_li, pred_li)
            mse = mean_squared_error(true_li, pred_li)
            mae = mean_absolute_error(true_li, pred_li)
            max_err = max_error(true_li, pred_li)
            #计算排名
            true_rank = rankdata(true_li)
            pred_rank = rankdata(pred_li)
            tau = kendalltau(true_rank, pred_rank)[0]
            data['target'].append(target_name)
            data['mape'].append(mape)
            data['rmse'].append(rmse)
            data['mse'].append(mse)
            data['mae'].append(mae)
            data['max_err'].append(max_err)
            data['tau'].append(tau)

            # data['rmse'].append(f'{rmse:.4f}')
            # data['mse'].append(f'{mse:.4f}')
            # data['tau'].append(f'{tau: .4f}')
            tot_mape += mape
            tot_rmse += rmse
            tot_mse += mse
            tot_mae += mae
            tot_max_err += max_err
            tot_tau += tau

            pred_std = d.get('pred_std')
            if pred_std is not None:
                assert type(pred_std) is np.ndarray, f'{type(pred_std)}'
                pred_std = np.mean(pred_std)
                data['pred_std'].append(pred_std)
                tot_std += pred_std
        data['target'].append('tot/avg')
        data['mape'].append(tot_mape)
        data['rmse'].append(tot_rmse)
        data['mse'].append(tot_mse)
        data['mae'].append(tot_mae)
        data['max_err'].append(tot_max_err)
        data['tau'].append(tot_tau / len(points_dict))
        if 'pred_std' in data:
            data['pred_std'].append(tot_std / len(points_dict))
    except ValueError as v:
        saver.log_info(f'Error {v}')
        data = defaultdict(list)

    # data['rmse'].append(f'{tot_rmse:.4f}')
    # data['mse'].append(f'{tot_mse:.4f}')
    # data['tau'].append(f'{tot_tau / len(points_dict):.4f}')
    df = pd.DataFrame(data)
    pd.set_option('display.max_columns', None)
    if print_result:
        saver.log_info(num_data)
        saver.log_info(df.round(4))
    # exit()
    return df
    # exit()

def _model_forward_batch(model, data, code_f=None):
    """与旧版 Net.forward(self, data)、(data, code_f)、(data, ablation_mode=...) 均兼容。"""
    import inspect
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return model(data)
    if 'ablation_mode' in params:
        return model(data, ablation_mode=getattr(FLAGS, 'ablation_mode', 'all'))
    if 'code_f' in params:
        if code_f is None:
            raise TypeError(
                "model.forward 需要参数 code_f（与 test() 中 model(data, text_list) 一致）；"
                "Regression dump 已改为传入 text_list。"
            )
        return model(data, code_f)
    return model(data)


def _regression_pred_csv_columns():
    _target_list = FLAGS.target if isinstance(FLAGS.target, list) else [FLAGS.target]
    target_list = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t for t in _target_list]
    cols = ['split', 'kernel', 'file']
    for t in target_list:
        cols.append(f'{t}_pred')
        cols.append(f'{t}_true')
    return cols, target_list


def _collect_regression_pred_rows(model, graph_list, split_name):
    """与 test() 相同的多模态注入与前向；每个样本一行，供 regression 权重导出。"""
    import re
    if not graph_list:
        return []
    _, target_list = _regression_pred_csv_columns()
    TEXT_ROOT = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_dataset/code"
    AST_ROOT = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_datasets/code/ast_final_dataset_1"

    loader = DataLoader(graph_list, batch_size=FLAGS.batch_size, shuffle=False, pin_memory=True,
                        collate_fn=custom_collate)
    rows = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(loader, position=0, total=len(loader), file=sys.stdout,
                         desc=f'export pred {split_name}'):
            data = data.to(FLAGS.device)
            text_list, ast_list = [], []
            for j in range(len(data.file)):
                file_name = str(data.file[j])
                kernel_name = str(data.kernel[j])
                idx_match = re.search(r'\d+', file_name)
                cdfg_id = int(idx_match.group()) if idx_match else None
                t_path = None
                if cdfg_id is not None:
                    for sub in ['poly', 'machsuite']:
                        p = os.path.join(TEXT_ROOT, sub, kernel_name, f"{cdfg_id + 1}.pt")
                        if os.path.exists(p):
                            t_path = p
                            break
                ast_id = cdfg_id
                a_path = os.path.join(AST_ROOT, f"{ast_id}.ast.pt") if cdfg_id is not None else ""
                try:
                    if t_path and os.path.exists(a_path):
                        t_feat = torch.load(t_path, map_location='cpu')
                        a_feat = torch.load(a_path, map_location='cpu')
                        text_list.append(t_feat if isinstance(t_feat, torch.Tensor) else torch.tensor(t_feat))
                        ast_list.append(a_feat if isinstance(a_feat, torch.Tensor) else torch.tensor(a_feat))
                    else:
                        text_list.append(torch.zeros(768))
                        ast_list.append(torch.zeros(784))
                except Exception:
                    text_list.append(torch.zeros(768))
                    ast_list.append(torch.zeros(784))
            data.text_feat = torch.stack(text_list).to(FLAGS.device)
            data.ast_feat = torch.stack(ast_list).to(FLAGS.device)

            out_dict, _, _ = _model_forward_batch(model, data, text_list)

            bs = len(data.kernel)
            for j in range(bs):
                row = {'split': split_name, 'kernel': str(data.kernel[j]), 'file': str(data.file[j])}
                for target_name in target_list:
                    model_key = 'perf' if target_name == 'actual_perf' else target_name
                    out = out_dict[model_key]
                    pred_val = out[j].item()
                    y_true = _get_y_with_target(data, target_name)
                    true_val = y_true[j].item()
                    if FLAGS.encode_log and target_name == 'actual_perf':
                        pred_val = 2 ** pred_val * (1 / FLAGS.normalizer)
                    row[f'{target_name}_pred'] = pred_val
                    row[f'{target_name}_true'] = true_val
                rows.append(row)
    return rows


def inference(dataset):   # 随机划分的数据集
    """
    重构后的推理函数：
    1. 使用与 train_main 相同的随机种子，自动识别出那 7 个 Inference Kernels
    2. 从 dataset.data_list 中筛选出这 7 个 Kernel 的全量样本
    3. 保持评估逻辑不变
    """
    from main import MACHSUITE_KERNEL, poly_KERNEL
    import numpy as np
    from numpy.random import permutation
    import torch
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import degree
    from collections import OrderedDict
    import os
    export_dir = os.path.dirname(__file__)

    # --- 1. 使用与训练时完全一致的 Kernel 划分逻辑 ---
    all_kernels = MACHSUITE_KERNEL + poly_KERNEL  # 共 22 个
    num_train_kernels = 15

    # 必须使用与 train_main 相同的种子 128
    np.random.seed(64)
    shuffled_indices = permutation(len(all_kernels))

    # 选出那 7 个被排除在训练之外的 Inference Kernels
    inference_kernel_names = [all_kernels[i] for i in shuffled_indices[num_train_kernels:]]

    saver.info(f'Detected {len(inference_kernel_names)} inference kernels: {inference_kernel_names}')

    # --- 2. 筛选样本 ---
    test_graphs = []
    for data in dataset.data_list:
        # 兼容性获取 kernel 属性名
        k_name = getattr(data, 'kernel', getattr(data, 'gname', 'N/A'))
        if k_name in inference_kernel_names:
            test_graphs.append(data)

    if len(test_graphs) == 0:
        sample_kernel = getattr(dataset.data_list[0], 'kernel', 'N/A') if len(dataset.data_list) > 0 else "None"
        saver.error(f"推理筛选结果为空！数据集中第一个 kernel 为: {sample_kernel}")
        raise ValueError(f"没有匹配到指定的推理 Kernel 样本。请检查 Data 对象属性。")

    # 与 inference 互补的 15 个训练 Kernel 上的样本（用于 regression 权重下的 train 预测导出）
    train_kernel_names = set(all_kernels[i] for i in shuffled_indices[:num_train_kernels])
    train_graphs = []
    for data in dataset.data_list:
        k_name = getattr(data, 'kernel', getattr(data, 'gname', 'N/A'))
        if k_name in train_kernel_names:
            train_graphs.append(data)

    # 导出当前 inference 数据池到 Excel（用于排查/对齐口径）
    # 输出包含：kernel、file，以及每个目标的真实值（与 test() 中 y_true 对齐的 target_list）
    try:
        _target_list = FLAGS.target if isinstance(FLAGS.target, list) else [FLAGS.target]
        target_list = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t for t in _target_list]

        export_rows = []
        for idx, d in enumerate(test_graphs):
            row = {
                'idx': idx,
                'kernel': getattr(d, 'kernel', getattr(d, 'gname', 'N/A')),
                'file': str(getattr(d, 'file', getattr(d, 'gfile', 'N/A'))),
            }
            for target_name in target_list:
                try:
                    y_true = _get_y_with_target(d, target_name).view(-1)
                    row[target_name] = float(y_true.mean().item()) if y_true.numel() > 0 else 0.0
                except Exception:
                    row[target_name] = float('nan')
            export_rows.append(row)

        export_df = pd.DataFrame(export_rows)
        out_xlsx = join(export_dir, 'inference_pool.xlsx')
        export_df.to_excel(out_xlsx, index=False)
        saver.info(f'[Export] inference pool exported to: {out_xlsx}')
    except Exception as e:
        # 兜底：不保证环境一定有 openpyxl/xlsxwriter
        saver.log_info(f'[Export] Excel export failed, fallback to CSV. Reason: {e}')
        try:
            _target_list = FLAGS.target if isinstance(FLAGS.target, list) else [FLAGS.target]
            target_list = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t for t in _target_list]

            export_rows = []
            for idx, d in enumerate(test_graphs):
                row = {
                    'idx': idx,
                    'kernel': getattr(d, 'kernel', getattr(d, 'gname', 'N/A')),
                    'file': str(getattr(d, 'file', getattr(d, 'gfile', 'N/A'))),
                }
                for target_name in target_list:
                    try:
                        y_true = _get_y_with_target(d, target_name).view(-1)
                        row[target_name] = float(y_true.mean().item()) if y_true.numel() > 0 else 0.0
                    except Exception:
                        row[target_name] = float('nan')
                export_rows.append(row)

            export_df = pd.DataFrame(export_rows)
            out_csv = join(export_dir, 'inference_pool.csv')
            export_df.to_csv(out_csv, index=False)
            saver.info(f'[Export] inference pool exported to CSV: {out_csv}')
        except Exception as e2:
            saver.error(f'[Export] CSV export also failed: {e2}')

    saver.log_info(f'Total Inference samples from {len(inference_kernel_names)} kernels: {len(test_graphs)}')

    # --- 3. 准备 DataLoader ---
    test_loader = DataLoader(test_graphs, batch_size=FLAGS.batch_size, pin_memory=True, collate_fn=custom_collate)

    # --- 4. 计算度分布 (PNA 初始化需要；含 train+inference 图，与训练时口径一致) ---
    all_deg_graphs = train_graphs + test_graphs
    max_degree = -1
    for data in all_deg_graphs:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in all_deg_graphs:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    # --- 5. 初始化模型结构 ---
    in_channels = 153  # 应与训练时保持一致
    if FLAGS.comparative_if and FLAGS.comparative_model == "pna":
        model = Net(in_channels=in_channels, target_list=FLAGS.target, deg=deg, task='regression').to(FLAGS.device)
    else:
        model = Net(in_channels=in_channels, target_list=FLAGS.target, task='regression').to(FLAGS.device)

    # --- 6. 加载权重：若存在 regression_model_state_dict.pth，先用其导出 train/test 全量预测到 data/，再恢复 FLAGS 指定权重 ---
    model_path = FLAGS.model_path if FLAGS.model_path else join(get_root_path(),
                                                                'save_models_and_data/best_model_state_dict.pth')
    regression_ckpt = join(get_root_path(), 'save_models_and_data', 'regression_model_state_dict.pth')
    data_out_dir = join(get_root_path(), 'data')
    os.makedirs(data_out_dir, exist_ok=True)
    data_dir_pred_csv_ok = False
    saver.log_info(
        f'[Regression dump] data_out_dir (abs)={os.path.abspath(data_out_dir)}, '
        f'regression_ckpt (abs)={os.path.abspath(regression_ckpt)}, exists={os.path.isfile(regression_ckpt)}'
    )

    if not os.path.exists(model_path):
        saver.error(f'Model path {model_path} does not exist!')
        raise RuntimeError("Inference requires a pre-trained model.")

    if os.path.isfile(regression_ckpt):
        model.load_state_dict(torch.load(regression_ckpt, map_location=torch.device(FLAGS.device)))
        saver.info(f'[Regression dump] Loaded {regression_ckpt} for full train/inference-kernel test predictions')
        import time
        ts = time.strftime('%Y%m%d_%H%M%S')
        pred_cols, _ = _regression_pred_csv_columns()
        p_tr_reg = join(data_out_dir, f'regression_model_pred_train_{ts}.csv')
        p_te_reg = join(data_out_dir, f'regression_model_pred_test_{ts}.csv')
        try:
            train_rows = _collect_regression_pred_rows(model, train_graphs, 'train')
            test_rows_dump = _collect_regression_pred_rows(model, test_graphs, 'test')
            df_tr = pd.DataFrame(train_rows) if train_rows else pd.DataFrame(columns=pred_cols)
            df_te = pd.DataFrame(test_rows_dump) if test_rows_dump else pd.DataFrame(columns=pred_cols)
            df_tr.to_csv(p_tr_reg, index=False)
            df_te.to_csv(p_te_reg, index=False)
            if os.path.isfile(p_tr_reg) and os.path.isfile(p_te_reg):
                data_dir_pred_csv_ok = True
            saver.info(
                f'[Regression dump] Saved {len(train_rows)} train / {len(test_rows_dump)} test rows to {data_out_dir}'
            )
        except Exception as e:
            saver.error(f'[Regression dump] failed: {e}')
    else:
        saver.log_info(f'[Regression dump] Skip (file not found): {regression_ckpt}')

    model.load_state_dict(torch.load(model_path, map_location=torch.device(FLAGS.device)))
    saver.info(f'Successfully loaded model from {model_path}')

    # 无 regression_model_state_dict.pth 时仍导出预测/真值（使用上面刚加载的 model_path 权重），避免 data/ 下始终无文件
    if not os.path.isfile(regression_ckpt):
        import time
        ts_fb = time.strftime('%Y%m%d_%H%M%S')
        pred_cols_fb, _ = _regression_pred_csv_columns()
        p_train = join(data_out_dir, f'inference_model_pred_train_{ts_fb}.csv')
        p_test = join(data_out_dir, f'inference_model_pred_test_{ts_fb}.csv')
        try:
            train_rows_fb = _collect_regression_pred_rows(model, train_graphs, 'train')
            test_rows_fb = _collect_regression_pred_rows(model, test_graphs, 'test')
            df_tr_fb = pd.DataFrame(train_rows_fb) if train_rows_fb else pd.DataFrame(columns=pred_cols_fb)
            df_te_fb = pd.DataFrame(test_rows_fb) if test_rows_fb else pd.DataFrame(columns=pred_cols_fb)
            df_tr_fb.to_csv(p_train, index=False)
            df_te_fb.to_csv(p_test, index=False)
            if os.path.isfile(p_train) and os.path.isfile(p_test):
                data_dir_pred_csv_ok = True
            saver.info(
                f'[Pred export] (fallback, weights={model_path}) wrote {len(train_rows_fb)} train / '
                f'{len(test_rows_fb)} test rows -> {os.path.abspath(p_train)} | {os.path.abspath(p_test)}'
            )
        except Exception as e:
            saver.error(f'[Pred export] fallback CSV failed: {e}')

    if not data_dir_pred_csv_ok:
        _warn = (
            '[inference] 未在 data 目录下成功生成 train/test 预测 CSV（请检查：是否存在 regression_model_state_dict.pth '
            '且 regression 导出是否报错；若无该权重则检查 fallback 的 [Pred export] 是否失败）。'
            f' 期望目录: {os.path.abspath(data_out_dir)}'
        )
        print(_warn)
        saver.log_info(_warn)

    # --- 7. 执行评估 ---
    # 注意：确保你的 test 函数返回 4 个值（包含 all_kernels 列表）
    testr, loss_dict, points_dict, all_kernels_ret = test(test_loader, 'test', model, 0, plot_test=True)

    # loss_dict / testr 来自 model：默认 FLAGS.loss=RMSE 时为各 batch 内 sqrt(MSE) 再对 batch 平均，与表格里 sklearn 的「全局 MSE」不是同一量；应对照 RMSE 列而非 MSE 列。
    saver.log_info(
        f'Avg total (per batch: sum of per-target {FLAGS.loss}; mean over batches): {testr:.7f}'
    )
    saver.log_info(f'Per-target avg {FLAGS.loss} (training criterion): {loss_dict}')

    # --- 8. 按单独 Kernel 打印报告 ---
    unique_kernels = sorted(list(set(all_kernels_ret)))
    # 导出到 Excel：保存每个 kernel / 每个 target 的误差指标
    export_rows = []
    for kname in unique_kernels:
        k_points = OrderedDict()
        for target in points_dict.keys():
            k_points[target] = {'true': [], 'pred': []}
            for idx, k_val in enumerate(all_kernels_ret):
                if k_val == kname:
                    k_points[target]['true'].append(points_dict[target]['true'][idx])
                    k_points[target]['pred'].append(points_dict[target]['pred'][idx])

        # 保持原有打印行为
        _report_rmse_etc(k_points, f"Kernel Performance Report: {kname}")

        # 同时收集可导出的指标
        kernel_results = _report_rmse_etc(k_points, f"Kernel Performance Report: {kname}", print_result=False)
        for target_name, metrics in kernel_results.items():
            export_rows.append({
                'kernel': kname,
                'target': target_name,
                'mse': metrics.get('mse'),
                'rmse': metrics.get('rmse'),
                'mae': metrics.get('mae'),
                'mape': metrics.get('mape'),
                'tau': metrics.get('tau'),
                'max_err': metrics.get('max_err'),
            })

    # 导出整体指标（kernel=ALL）
    overall_results = _report_rmse_etc(points_dict, "Overall Inference Metrics", print_result=False)
    for target_name, metrics in overall_results.items():
        export_rows.append({
            'kernel': 'ALL',
            'target': target_name,
            'mse': metrics.get('mse'),
            'rmse': metrics.get('rmse'),
            'mae': metrics.get('mae'),
            'mape': metrics.get('mape'),
            'tau': metrics.get('tau'),
            'max_err': metrics.get('max_err'),
        })

    # 写入 Excel
    try:
        import time
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_path = join(export_dir, f'inference_metrics_{ts}.xlsx')
        df = pd.DataFrame(export_rows)
        # 优先使用 xlsxwriter，避免 openpyxl 缺失导致导出失败
        df.to_excel(out_path, index=False, engine='xlsxwriter')
        saver.info(f'[Excel Export] inference metrics saved to: {out_path}')
    except Exception as e:
        saver.error(f'[Excel Export] failed: {e} (fallback to CSV)')
        try:
            import time
            ts = time.strftime('%Y%m%d_%H%M%S')
            out_csv = join(export_dir, f'inference_metrics_{ts}.csv')
            df = pd.DataFrame(export_rows)
            df.to_csv(out_csv, index=False)
            saver.info(f'[CSV Export] inference metrics saved to: {out_csv}')
        except Exception as e2:
            saver.error(f'[CSV Export] failed: {e2}')

    return testr, loss_dict, points_dict



def train_main(dataset, pragma_dim=None):     # 随机划分数据集（不按 kernel 过滤 all-zero 配置）
    saver.info(f'开始读取数据集并执行按 Kernel 归纳式（Inductive）划分')
    from main import MACHSUITE_KERNEL, poly_KERNEL
    import numpy as np
    from numpy.random import permutation
    import torch
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import degree
    from os.path import join
    from utils import get_root_path, OurTimer
    # --- 1. Kernel 级别的彻底切分 ---
    all_kernels = MACHSUITE_KERNEL + poly_KERNEL
    num_train_kernels = 15
    np.random.seed(64)
    shuffled_indices = permutation(len(all_kernels))
    train_kernel_names = [all_kernels[i] for i in shuffled_indices[:num_train_kernels]]
    inference_kernel_names = [all_kernels[i] for i in shuffled_indices[num_train_kernels:]]
    saver.info(f'训练/验证 Kernel (15个): {train_kernel_names}')
    saver.info(f'推理 (Inference) Kernel (7个): {inference_kernel_names}')
    # --- 2. 样本池分类 ---
    train_val_pool = []
    test_list = []
    for data in dataset.data_list:
        k_name = getattr(data, 'kernel', getattr(data, 'gname', 'N/A'))
        if k_name in train_kernel_names:
            train_val_pool.append(data)
        elif k_name in inference_kernel_names:
            test_list.append(data)
    if len(train_val_pool) == 0:
        raise ValueError("错误：未能在数据集中匹配到 15 个训练 Kernel 的样本，请检查 data 对象的属性名是否为 'kernel'")
    # --- 3. 在训练池内部进行 8:2 划分 (Train/Val) ---
    num_tv = len(train_val_pool)
    l_t = int(num_tv * 0.8)
    tv_indices = permutation(num_tv)
    train_list = [train_val_pool[j] for j in tv_indices[:l_t]]
    val_list = [train_val_pool[j] for j in tv_indices[l_t:]]
    saver.log_info(f'划分完成统计：')
    saver.log_info(f'- 训练集 (Train): {len(train_list)} 样本 (来自 {len(train_kernel_names)} 个 Kernel)')
    saver.log_info(f'- 验证集 (Val):   {len(val_list)} 样本 (来自 {len(train_kernel_names)} 个 Kernel)')
    saver.log_info(f'- 测试集 (Test):  {len(test_list)} 样本 (来自 {len(inference_kernel_names)} 个全新 Kernel)')
    # --- 4. 构建 DataLoader ---
    train_loader = DataLoader(train_list, batch_size=FLAGS.batch_size, shuffle=True,
                              pin_memory=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_list, batch_size=FLAGS.batch_size,
                            pin_memory=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_list, batch_size=FLAGS.batch_size,
                             pin_memory=True, collate_fn=custom_collate)
    # --- 5. 计算训练集度分布 (PNA 模型必须) ---
    max_degree = -1
    for data in train_list:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_list:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    # --- 6. 模型初始化 ---
    in_channels = 153
    if FLAGS.comparative_if and FLAGS.comparative_model == "pna":
        model = Net(in_channels=in_channels, target_list=FLAGS.target, deg=deg, task='regression').to(FLAGS.device)
    else:
        model = Net(in_channels=in_channels, target_list=FLAGS.target, task='regression').to(FLAGS.device)
    if FLAGS.model_path is not None:
        model.load_state_dict(torch.load(FLAGS.model_path, map_location=torch.device(FLAGS.device)))
        saver.info(f'Loaded model from {FLAGS.model_path}')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # --- 7. 训练循环 ---
    train_losses, val_losses, test_losses = [], [], []
    lm = 1e7  # 记录最小 val loss
    for epoch in range(FLAGS.epoch_num):
        timer = OurTimer()
        loss, loss_dict_train = train(epoch, model, train_loader, optimizer)
        val_loss, loss_dict_val, _ = test(val_loader, 'val', model, epoch)
        test_loss, loss_dict_test, _ = test(test_loader, 'test', model, epoch)
        saver.log_info(
            f'Epoch: {epoch + 1:03d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, Time: {timer.time_and_clear()}'
        )
        train_losses.append(loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        # 保存验证/测试标准你按自己需求选
        if test_loss < lm:
            torch.save(model.state_dict(), join(get_root_path(), 'save_models_and_data/regression_model_state_dict.pth'))
            lm = test_loss
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_losses, 'g', label='Train')
    plt.plot(val_losses, 'b', label='Val')
    plt.plot(test_losses, 'r', label='Test (Unseen Kernels)')
    plt.legend()
    plt.savefig(join(saver.get_log_dir(), 'losses.png'))
    torch.save(model.state_dict(), join(get_root_path(), 'save_models_and_data/regression_model_final.pth'))

import re
def train(epoch, model, train_loader, optimizer):           # 3特征融合的train
    model.train()

    # --- 1. 定义三模态特征根目录 (请确保路径准确) ---
    TEXT_ROOT = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_dataset/code"
    AST_ROOT = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_datasets/code/ast_final_dataset_1"

    total_loss = 0
    _target_list = FLAGS.target
    if not isinstance(FLAGS.target, list):
        _target_list = [FLAGS.target]

    target_list = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t for t in _target_list]
    loss_dict_accum = {t: 0.0 for t in target_list}

    # 开始 Batch 循环
    for data in tqdm(train_loader, position=0, total=len(train_loader), file=sys.stdout,
                     desc=f"Epoch {epoch + 1} Train"):

        # 将图数据移动到 GPU (x, edge_index 等)
        data = data.to(FLAGS.device)

        # --- 2. 手动特征注入逻辑 (核心修改) ---
        text_list = []
        ast_list = []

        # 遍历当前 Batch 中的所有样本文件名和 Kernel 名
        # 注意：DataBatch 模式下 data.file 和 data.kernel 是列表
        for j in range(len(data.file)):
            file_name = str(data.file[j])  # 如 'data_1.pt'
            kernel_name = str(data.kernel[j])  # 如 'gemm-ncubed'

            # A. 提取编号 (从 'data_1.pt' 提取数字 1)
            idx_match = re.search(r'\d+', file_name)
            # if not idx_match:
            #     # 异常处理：找不到编号则补 0 向量
            #     text_list.append(torch.zeros(768))
            #     ast_list.append(torch.zeros(784))
            #     continue

            cdfg_id = int(idx_match.group())

            # B. 匹配 Text 路径 (Text 的 2.pt 对应 data_1.pt)
            t_path = None
            for sub in ['poly', 'machsuite']:
                p = os.path.join(TEXT_ROOT, sub, kernel_name, f"{cdfg_id+1}.pt")
                if os.path.exists(p):
                    t_path = p
                    break

            # C. 匹配 AST 路径 (AST 的 0.ast.pt 对应 data_1.pt，即偏移 -1)
            ast_id = cdfg_id
            a_path = os.path.join(AST_ROOT, f"{ast_id}.ast.pt")

            # D. 加载数据并存入列表
            try:
                if t_path and os.path.exists(a_path):
                    t_feat = torch.load(t_path, map_location='cpu')
                    a_feat = torch.load(a_path, map_location='cpu')
                    # 确保是 Tensor 格式
                    text_list.append(t_feat if isinstance(t_feat, torch.Tensor) else torch.tensor(t_feat))
                    ast_list.append(a_feat if isinstance(a_feat, torch.Tensor) else torch.tensor(a_feat))
                else:
                    # 路径不通，补 0 占位
                    text_list.append(torch.zeros(768))
                    ast_list.append(torch.zeros(784))
            except Exception:
                text_list.append(torch.zeros(768))
                ast_list.append(torch.zeros(784))

        # E. 堆叠 Tensor 并移动到 GPU 挂载到 data 对象
        # 这一步强制补齐了 model.py 中 forward 需要的两个 key
        data.text_feat = torch.stack(text_list).to(FLAGS.device)
        data.ast_feat = torch.stack(ast_list).to(FLAGS.device)

        # 1. 梯度清零
        optimizer.zero_grad()

        # 2. 模型推理 (此时 data 已经自带了 text_feat 和 ast_feat)
        # out_dict, loss, loss_dict_batch = model(data)         # 老版本，未加入消融实验
        out_dict, loss, loss_dict_batch = _model_forward_batch(model, data, text_list)    # MPM：forward(data, code_f)

        # 3. 反向传播和优化
        loss.backward()
        optimizer.step()

        # 4. 统计 Loss
        total_loss += loss.item() * data.num_graphs
        for t in target_list:
            model_key = 'perf' if t == 'actual_perf' else t
            if model_key in loss_dict_batch:
                loss_dict_accum[t] += loss_dict_batch[model_key].item()

    # --- 训练结束，计算平均值 ---
    avg_loss = total_loss / len(train_loader.dataset)
    avg_loss_dict = {key: v / len(train_loader) for key, v in loss_dict_accum.items()}

    print(f"\n>> [Epoch {epoch + 1} Train] Total Avg Loss: {avg_loss:.8e}")
    for k, v in avg_loss_dict.items():
        print(f"   - {k:15s} Loss: {v:.8e}")

    return avg_loss, avg_loss_dict


def inference_loss_function(pred, true):
    return (pred - true) ** 2

def test(loader, tvt, model, epoch, plot_test=False, test_losses=[-1], design_points=None):     # 3特征融合使用的test
    """
    完整版测试函数：集成三模态特征手动加载、Loss统计、预测点收集及绘图功能
    """
    model.eval()

    # --- 1. 定义特征根目录 (必须与 train 保持绝对一致) ---
    TEXT_ROOT = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_dataset/code"
    AST_ROOT = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_datasets/code/ast_final_dataset_1"

    total_loss = 0
    loss_dict = {}
    points_dict = OrderedDict()

    all_kernels = []        # 为了打印每个kernel的rmse等信息而添加的

    # 处理目标列表
    _target_list = FLAGS.target
    if not isinstance(FLAGS.target, list):
        _target_list = [FLAGS.target]

    target_list = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t for t in _target_list]

    for t in target_list:
        loss_dict[t] = 0.0
        points_dict[t] = {'true': [], 'pred': []}

    # 开始验证/测试循环
    with torch.no_grad():
        for data in tqdm(loader, position=0, total=len(loader), file=sys.stdout,
                         desc=f"Epoch {epoch + 1} {tvt}"):

            # 将基础图数据移动到设备
            data = data.to(FLAGS.device)
            all_kernels.extend(data.kernel)         #  为了打印每个kernel的rmse等信息而添加的

            # --- 2. 手动特征注入逻辑 (核心修改) ---
            text_list = []
            ast_list = []

            # 遍历 Batch 内的每个样本 (data.file 是包含文件名的列表)
            for j in range(len(data.file)):
                file_name = str(data.file[j])  # 如 'data_1.pt'
                kernel_name = str(data.kernel[j])  # 如 'gemm-ncubed'

                # A. 提取编号 (data_1.pt -> 1)
                idx_match = re.search(r'\d+', file_name)
                cdfg_id = int(idx_match.group()) if idx_match else None

                # B. 匹配 Text 路径 (1.pt 对应 data_1.pt)
                t_path = None
                if cdfg_id is not None:
                    for sub in ['poly', 'machsuite']:
                        p = os.path.join(TEXT_ROOT, sub, kernel_name, f"{cdfg_id+1}.pt")
                        if os.path.exists(p):
                            t_path = p
                            break

                # C. 匹配 AST 路径 (0.ast.pt 对应 data_1.pt, 偏移 -1)
                # ast_id = cdfg_id - 1 if cdfg_id is not None else -1
                ast_id = cdfg_id
                a_path = os.path.join(AST_ROOT, f"{ast_id}.ast.pt")

                # D. 加载特征并处理异常情况 (补0向量)
                try:
                    if t_path and os.path.exists(a_path):
                        t_feat = torch.load(t_path, map_location='cpu')
                        a_feat = torch.load(a_path, map_location='cpu')
                        text_list.append(t_feat if isinstance(t_feat, torch.Tensor) else torch.tensor(t_feat))
                        ast_list.append(a_feat if isinstance(a_feat, torch.Tensor) else torch.tensor(a_feat))
                    else:
                        text_list.append(torch.zeros(768))
                        ast_list.append(torch.zeros(784))
                except Exception:
                    text_list.append(torch.zeros(768))
                    ast_list.append(torch.zeros(784))

            # E. 堆叠特征并挂载到 data 对象 (关键：供模型 fusion_block 使用)
            data.text_feat = torch.stack(text_list).to(FLAGS.device)
            data.ast_feat = torch.stack(ast_list).to(FLAGS.device)

            # --- 3. 模型推理 ---
            # out_dict, loss, loss_dict_ 结构需与 model.forward 对应
            # out_dict, loss, loss_dict_ = model(data)      # 老版本，未加入消融实验的
            out_dict, loss, loss_dict_ = model(data, ablation_mode=FLAGS.ablation_mode)     # 加入消融实验而进行的改进


            total_loss += loss.item()
            for t in target_list:
                if t in loss_dict_:
                    loss_dict[t] += loss_dict_[t].item()

            # --- 4. 收集预测值与真实值 ---
            for target_name in target_list:
                model_key = 'perf' if target_name == 'actual_perf' else target_name
                out = out_dict[model_key]
                y_true = _get_y_with_target(data, target_name)

                for i in range(len(out)):
                    pred_val = out[i].item()
                    true_val = y_true[i].item()

                    # 针对 perf 的对数反转逻辑
                    if FLAGS.encode_log and target_name == 'actual_perf':
                        pred_val = 2 ** (pred_val) * (1 / FLAGS.normalizer)

                    points_dict[target_name]['pred'].append(pred_val)
                    points_dict[target_name]['true'].append(true_val)

    # --- 5. 计算统计结果并打印 ---
    avg_loss = total_loss / len(loader)
    avg_loss_dict = {key: v / len(loader) for key, v in loss_dict.items()}

    print(f"\n>> [Epoch {epoch + 1} {tvt}] Avg objective (batch sum of per-target {FLAGS.loss}, / batches): {avg_loss:.8e}")
    for k, v in avg_loss_dict.items():
        print(f"   - {k:10s} avg {FLAGS.loss}: {v:.8e}")

    # --- 6. 绘图逻辑 ---
    if FLAGS.plot_pred_points and tvt == 'test' and (plot_test or (test_losses and avg_loss < min(test_losses))):
        from utils import plot_points, plot_points_with_subplot
        if not FLAGS.multi_target:
            target_0 = FLAGS.target[0]
            plot_points({f'{target_0}-pred_points': points_dict[target_0]['pred'],
                         f'{target_0}-true_points': points_dict[target_0]['true']},
                        f'epoch_{epoch + 1}_{tvt}', saver.get_log_dir())
        else:
            plot_points_with_subplot(points_dict, f'epoch_{epoch + 1}_{tvt}', saver.get_log_dir(), target_list)

    # --- 7. 推理任务评估 ---
    if FLAGS.subtask == 'inference':
        _report_rmse_etc(points_dict, f'epoch {epoch}:', True)

    return avg_loss, avg_loss_dict, points_dict ,all_kernels   #  all_kernels为了打印每个kernel的rmse等信息而添加的
    