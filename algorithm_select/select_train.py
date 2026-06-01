import os
import glob

from numpy.random import permutation
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.config import FLAGS, TARGETS
import sys
from src.saver import saver
from src.utils import MLP, OurTimer, get_save_path, _get_y_with_target, get_root_path
from torch_geometric.utils import degree
from src.model import Net
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
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
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
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'tau': tau,
            'max_err': max_err
        }

        if print_result:
            # 使用科学计数法打印，方便查看 10^10 级别的 Loss
            saver.log_info(
                f"Target: {target_name:15s} | RMSE: {rmse:.4e} | MAE: {mae:.4e} | MAPE: {mape:.4f} | Tau: {tau:.4f}")

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


def compute_causal_loss(
    model, data, code, design_points: Optional[List[DesignPoint]],
    target_list: List[str], flags, out_dict=None
) -> Optional[torch.Tensor]:
    """
    计算因果损失 L_causal = ||Δŷ - Δy||²

    Args:
        model: Net 模型
        data: PyTorch Geometric Batch 对象
        code: 代码特征列表
        design_points: List of DesignPoint (与 batch 中的样本对应)
        target_list: 目标列表
        flags: FLAGS 对象
        out_dict: 模型的输出字典（包含预测值），如果为 None 则从模型中获取

    Returns:
        causal_loss: Tensor scalar，如果无法计算则返回 None
    """
    if design_points is None or len(design_points) == 0:
        return None

    if not flags.use_causal:
        return None

    # 过滤掉 None 的 design_points
    valid_indices = [i for i, dp in enumerate(design_points) if dp is not None]
    if len(valid_indices) < 2:
        return None

    # 获取 batch 信息
    from torch_geometric.data import Batch
    if not isinstance(data, Batch):
        return None

    batch_size = data.num_graphs if hasattr(data, 'num_graphs') else len(design_points)
    if batch_size < 2:
        return None

    # 使用 out_dict（来自模型 forward 的输出）
    if out_dict is None:
        return None

    # 获取真实的 QoR 值
    true_values = {}
    for target in target_list:
        y = _get_y_with_target(data, target)
        if y is not None:
            if flags.task == 'regression':
                true_values[target] = y.view((len(y), flags.out_dim))
            else:
                true_values[target] = y.view((len(y)))
        else:
            return None

    # 在当前 batch 内创建配对
    # 限制：Hamming distance <= 2（最多 2 个 pragma 不同）
    valid_design_points = [(i, design_points[i]) for i in valid_indices]
    pairs = []

    for i in range(len(valid_design_points)):
        for j in range(i + 1, len(valid_design_points)):
            idx1, dp1 = valid_design_points[i]
            idx2, dp2 = valid_design_points[j]

            # 计算 Hamming 距离
            dist = hamming_distance(dp1, dp2)
            if 1 <= dist <= 2:  # 限制在 1-2 个 pragma 不同
                pairs.append((idx1, idx2))

    if len(pairs) == 0:
        return None

    # 限制配对数量
    max_pairs_per_batch = getattr(flags, 'causal_max_pairs_per_batch', 10)
    if len(pairs) > max_pairs_per_batch:
        import random
        pairs = random.sample(pairs, max_pairs_per_batch)

    # 计算因果损失
    device = next(model.parameters()).device
    total_causal_loss = 0.0
    valid_pairs = 0

    try:
        for idx1, idx2 in pairs:
            # 计算预测差异 Δŷ = ŷ(p2) - ŷ(p1)
            # 计算真实差异 Δy = y(p2) - y(p1)
            pred_diff = {}
            true_diff = {}

            for target in target_list:
                if target not in out_dict or target not in true_values:
                    continue

                # 使用 out_dict 中的张量，但通过 detach() 断开计算图
                # 原因：out_dict 中的张量已经参与了主损失的计算，如果再次用于因果损失，
                # 会导致计算图被重复使用，从而在 backward() 时出错
                # 注意：detach() 后，因果损失的梯度不会回传到预测值，但会通过主损失回传到模型参数
                pred1 = out_dict[target][idx1:idx1+1].detach()  # 保持维度，断开计算图
                pred2 = out_dict[target][idx2:idx2+1].detach()
                true1 = true_values[target][idx1:idx1+1]
                true2 = true_values[target][idx2:idx2+1]

                # 计算差异（这些操作会创建新的计算图，但不会与主损失的计算图重叠）
                pred_diff[target] = pred2 - pred1
                true_diff[target] = true2 - true1

            if len(pred_diff) == 0:
                continue

            # 计算每个目标的损失，然后平均
            target_losses = []
            for target in target_list:
                if target in pred_diff and target in true_diff:
                    # L_causal = ||Δŷ - Δy||²
                    diff = pred_diff[target] - true_diff[target]
                    if flags.task == 'regression' and flags.out_dim > 1:
                        # 多维度情况，计算所有维度的 MSE
                        loss_target = torch.mean(diff ** 2)
                    else:
                        loss_target = torch.mean(diff ** 2)
                    target_losses.append(loss_target)

            if len(target_losses) > 0:
                pair_loss = torch.stack(target_losses).mean()
                total_causal_loss += pair_loss
                valid_pairs += 1

        if valid_pairs == 0:
            return None

        # 返回平均因果损失
        return total_causal_loss / valid_pairs

    except Exception as e:
        # 如果计算失败，返回 None（不影响主训练流程）
        return None





def inference(dataset, mode, exclude_fold_idx):
    """
    重构后的推理函数：
    1. 直接从 dataset.data_list 获取全量数据进行筛选
    2. 筛选出训练集未见过的 Kernels (MachSuite[5:] 和 Poly[10:])
    3. 保持 PNA 初始化、模型加载和评估逻辑不变
    """
    saver.info(f'正在启动训练: , 折数={exclude_fold_idx}')


    exclude_kernels = set(K_FOLDS[exclude_fold_idx])

    li_g = []
    for data in dataset.data_list:
        # 兼容性检查：获取 kernel 名称
        k_name = getattr(data, 'kernel', 'N/A')
        # 如果数据中没有 kernel 属性，尝试从文件名解析 (假设文件名开头是 kernel 名)
        if k_name == 'N/A' and hasattr(data, 'file'):
            k_name = data.file.split('_')[0]

        if k_name in exclude_kernels:
            li_g.append(data)

    if len(li_g) == 0:
        saver.error(f"筛选结果为空！Fold {exclude_fold_idx} 排除掉的 Kernel 可能是全部数据。")
        return

    test_graphs = li_g
    # 3. 准备 DataLoader
    test_loader = DataLoader(test_graphs, batch_size=FLAGS.batch_size, pin_memory=True, collate_fn=custom_collate)

    # 4. PNA 必需的参数计算 (edge_dim 和 deg)
    # 计算度分布
    max_degree = -1
    for data in test_graphs:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in test_graphs:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    # 5. 初始化模型结构
    in_channels = test_graphs[0].x.shape[1]

    if FLAGS.comparative_if:
        if FLAGS.comparative_model == "pna":
            model = Net(
                in_channels=in_channels,
                target_list=FLAGS.target,
                deg=deg,
                task='regression'
            ).to(FLAGS.device)
        else:
            model = Net(
                in_channels=in_channels,
                target_list=FLAGS.target,
                task='regression'
            ).to(FLAGS.device)
    else:
        # 默认模型结构
        model = Net(
            in_channels=in_channels,
            target_list=FLAGS.target,
            task='regression'
        ).to(FLAGS.device)

    # 6. 加载预训练模型
    cv_weight_dir = join(get_root_path(), 'model_select_weight/cv_weights')
    final_save_path = join(cv_weight_dir, f"expert_{mode}_exclude_fold_{exclude_fold_idx}.pth")

    model_path = final_save_path
    if model_path is not None:
        old_state_dict = torch.load(model_path, map_location=torch.device(FLAGS.device))
        model.load_state_dict(old_state_dict)
        saver.info(f'Loaded model from {model_path}')
    else:
        saver.error(f'model path should be set during inference')
        raise RuntimeError("Missing model_path for inference")

    print(model)
    # saver.log_model_architecture(model)

    # 7. 执行测试评估
    # 调用 test 函数进行推理
    testr, loss_dict, points_dict = test(test_loader, 'test', model, 0, plot_test=True)

    saver.log_info(f'Inference loss breakdown: {loss_dict}')
    saver.log_info(f'Overall Inference loss: {testr:.7f}')

    return testr, loss_dict, points_dict

from fold_config import K_FOLDS, ALL_KERNEL


def train_main(dataset, mode, exclude_fold_idx):
    saver.info(f'正在启动训练: 模式={mode}, 排除折数={exclude_fold_idx}')


    exclude_kernels = set(K_FOLDS[exclude_fold_idx])

    li_g = []
    for data in dataset.data_list:
        # 兼容性检查：获取 kernel 名称
        k_name = getattr(data, 'kernel', 'N/A')
        # 如果数据中没有 kernel 属性，尝试从文件名解析 (假设文件名开头是 kernel 名)
        if k_name == 'N/A' and hasattr(data, 'file'):
            k_name = data.file.split('_')[0]

        if k_name not in exclude_kernels:
            li_g.append(data)

    if len(li_g) == 0:
        saver.error(f"筛选结果为空！Fold {exclude_fold_idx} 排除掉的 Kernel 可能是全部数据。")
        return

    # --- 3. 随机切分 (70% / 15% / 15%) ---
    li_len = len(li_g)
    l_t, l_v = int(li_len * 0.7), int(li_len * 0.15)
    rinx = permutation(range(li_len))

    train_list = [li_g[j] for j in rinx[0:l_t]]
    val_list = [li_g[j] for j in rinx[l_t: l_t + l_v]]
    test_list = [li_g[j] for j in rinx[l_t + l_v:]]

    print(f"DEBUG: 第一个样本的内容: {train_list[0]}")
    if not hasattr(train_list[0], 'file'):
        print("❌ 错误：你的 Data 对象里没有 'file' 属性，导致 custom_collate 无法匹配文件名！")

    saver.log_info(f'筛选完成！排除 {exclude_kernels} 后剩余 {li_len} 个样本。')
    saver.log_info(f'切分结果: {len(train_list)} train, {len(val_list)} val, {len(test_list)} test')

    # --- 4. 构建 DataLoader ---
    train_loader = DataLoader(train_list, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=True,
                              collate_fn=custom_collate)
    val_loader = DataLoader(val_list, batch_size=FLAGS.batch_size, pin_memory=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_list, batch_size=FLAGS.batch_size, pin_memory=True, collate_fn=custom_collate)


    # 4. 计算度分布 (PNA 专用)
    max_degree = -1
    for data in train_list:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_list:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    # 初始化模型
    if FLAGS.comparative_if:
        if FLAGS.comparative_model == "pna":
            model = Net(
                in_channels=784,  # 你的 AST/cdfg 特征维度 153 778  784
                target_list=FLAGS.target,  # 你的多目标列表 ['perf', 'util-LUT', ...]
                deg=deg,  # 刚才计算出来的 deg 张量
                task='regression'
            ).to(FLAGS.device)
        # model = Net(deg, edge_dim).to(FLAGS.device)
        else:
            model = Net(
                in_channels=784,  # 你的 AST/cdfg 特征维度   153 778  784
                target_list=FLAGS.target,  # 你的多目标列表 ['perf', 'util-LUT', ...]
                task='regression'
            ).to(FLAGS.device)
        # model = Net().to(FLAGS.device)
    else:
        model = Net(
            in_channels=784,  # 你的 AST 特征维度             153 778 784
            target_list=FLAGS.target,  # 你的多目标列表 ['perf', 'util-LUT', ...]
            task='regression'
        ).to(FLAGS.device)


    # if FLAGS.model_path != None:
    #     model.load_state_dict(torch.load(FLAGS.model_path, map_location=torch.device(FLAGS.device)))
    #     saver.info(f'loaded model from {FLAGS.model_path}')
    #
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 定义 CV 权重保存路径
    cv_weight_dir = join(get_root_path(), 'model_select_weight/cv_weights')
    if not os.path.exists(cv_weight_dir):
        os.makedirs(cv_weight_dir)
    final_save_path = join(cv_weight_dir, f"expert_{mode}_exclude_fold_{exclude_fold_idx}.pth")

    train_losses, val_losses, test_losses = [], [], []
    epochs = range(FLAGS.epoch_num)
    lm = 1e7  # 初始化最小 loss 记录

    # 开始训练循环
    for epoch in epochs:
        plot_test = False
        timer = OurTimer()
        saver.log_info(f'Epoch {epoch + 1} train')

        # 1. 训练
        loss, loss_dict_train = train(epoch, model, train_loader, optimizer)

        # 2. 验证
        if len(val_loader) > 0:
            saver.log_info(f'\nEpoch {epoch + 1} val')
            val, loss_dict_val, _ = test(val_loader, 'val', model, epoch)
            val_losses.append(val)
        else:
            val = loss  # 无验证集时

        # 3. 测试
        saver.log_info(f'\nEpoch {epoch + 1} test')
        testr, loss_dict_test, points_dict = test(test_loader, 'test', model, epoch, plot_test, test_losses)

        # 打印日志
        saver.log_info((f'\nTrain loss breakdown {loss_dict_train}'))
        saver.log_info((f'\nTest loss breakdown {loss_dict_test}'))
        if len(val_loader) > 0:
            saver.log_info((f'\nVal loss breakdown {loss_dict_val}'))
            saver.log_info(('Epoch: {:03d}, Train Loss: {:.4f}, Val loss: {:.4f}, Test: {:.4f}) Time: {}'.format(
                epoch + 1, loss, val, testr, timer.time_and_clear())))
        else:
            saver.log_info(('Epoch: {:03d}, Loss: {:.4f}, Train loss: {:.3f}, Test: {:.3f}) Time: {}'.format(
                epoch + 1, loss, loss, testr, timer.time_and_clear())))

        train_losses.append(loss)
        test_losses.append(testr)

        # 4. 保存最佳模型逻辑
        if loss < lm:
            torch.save(model.state_dict(), final_save_path)
            lm = loss

        # 5. 早停检查
        if len(train_losses) > 50:
            if len(set(train_losses[-50:])) == 1 and len(set(test_losses[-50:])) == 1:
                break

    # --- 开始画图部分 ---
        # --- 修正后的画图部分 ---
        import matplotlib
        # 彻底解决 headless 环境报错，改用 Agg 后端
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        final_epochs = range(len(train_losses))
        plt.figure()
        plt.plot(final_epochs, train_losses, 'g', label='Training loss')
        if len(val_loader) > 0:
            plt.plot(final_epochs, val_losses, 'b', label='Validation loss')
        plt.plot(final_epochs, test_losses, 'r', label='Testing loss')
        plt.title('Training, Validation, and Testing loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # 保存图片
        save_path = join(saver.get_log_dir(), 'losses.png')
        plt.savefig(save_path, bbox_inches='tight')

        # 核心修改：在 headless 环境下禁止调用 plt.show()，因为它会尝试打开窗口
        # plt.show()  <- 这一行要注释掉或删掉

        print(f"📈 训练曲线已保存至: {save_path}")

    del model
    torch.cuda.empty_cache()
    saver.info(f'✅ Fold {exclude_fold_idx} 训练完成，最佳权重已保存至: {final_save_path}')

    # 最终保存
    # if FLAGS.task == 'regression':
    #     torch.save(model.state_dict(), join(get_root_path(), 'save_models_and_data/regression_model_state_dict.pth'))


def train(epoch, model, train_loader, optimizer):       # 原来的train，没有进行3特征融合的train
    model.train()

    total_loss = 0
    _target_list = FLAGS.target
    if not isinstance(FLAGS.target, list):
        _target_list = [FLAGS.target]

    # 处理 actual_perf 的名称映射，确保和 model.py 里的 Key 一致
    target_list = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t for t in _target_list]

    # 初始化用于记录本轮平均 loss 的字典
    loss_dict_accum = {t: 0.0 for t in target_list}

    # 开始 Batch 循环
    for data in tqdm(train_loader, position=0, total=len(train_loader), file=sys.stdout,
                     desc=f"Epoch {epoch + 1} Train"):
        data = data.to(FLAGS.device)

        print(f"DEBUG: Batch data keys: {data.keys}")   # 三特征融合调试添加

        # 1. 梯度清零
        optimizer.zero_grad()

        # 2. 模型推理
        # 注意：这里的 out_dict 是字典，loss 是所有目标 loss 的总和
        out_dict, loss, loss_dict_batch = model(data)

        # 3. 反向传播和优化
        loss.backward()
        optimizer.step()

        # 4. 统计 Loss
        # total_loss 记录所有样本的总损失
        total_loss += loss.item() * data.num_graphs

        # 累加每个目标的 Loss（用于最后打印详细清单）
        for t in target_list:
            # 兼容处理：如果 target_list 里叫 actual_perf，但 model 返回里叫 perf
            model_key = 'perf' if t == 'actual_perf' else t
            if model_key in loss_dict_batch:
                loss_dict_accum[t] += loss_dict_batch[model_key].item()

    # --- 训练结束，计算平均值并输出 ---
    avg_loss = total_loss / len(train_loader.dataset)
    # 计算每个目标的平均 loss (按 Batch 数量平摊)
    avg_loss_dict = {key: v / len(train_loader) for key, v in loss_dict_accum.items()}

    # 这里就是你要求的：像 10^10 那样输出详细 Loss
    print(f"\n>> [Epoch {epoch + 1} Train] Total Avg Loss: {avg_loss:.8e}")
    for k, v in avg_loss_dict.items():
        print(f"   - {k:15s} Loss: {v:.8e}")

    return avg_loss, avg_loss_dict

def inference_loss_function(pred, true):
    return (pred - true) ** 2


# 进行test
def test(loader, tvt, model, epoch, plot_test=False, test_losses=[-1], design_points=None):     # 原来的test
    model.eval()
    total_loss = 0
    loss_dict = {}
    from collections import OrderedDict
    points_dict = OrderedDict()

    _target_list = FLAGS.target
    if not isinstance(FLAGS.target, list):
        _target_list = [FLAGS.target]

    target_list = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t for t in _target_list]

    for t in target_list:
        loss_dict[t] = 0.0
        points_dict[t] = {'true': [], 'pred': []}

    with torch.no_grad():
        for data in tqdm(loader, position=0, total=len(loader), file=sys.stdout, desc=f"Epoch {epoch + 1} {tvt}"):
            data = data.to(FLAGS.device)

            # out_dict 是字典 {'perf': tensor, 'util-LUT': tensor, ...}
            out_dict, loss, loss_dict_ = model(data)

            total_loss += loss.item()
            for t in target_list:
                if t in loss_dict_:
                    loss_dict[t] += loss_dict_[t].item()

            for target_name in target_list:
                # 从字典中获取对应目标的预测张量 [Batch, 1]
                # 兼容处理 actual_perf 的 Key 名
                model_key = 'perf' if target_name == 'actual_perf' else target_name
                out = out_dict[model_key]

                y_true = _get_y_with_target(data, target_name)

                # 遍历 Batch 内的每个样本存入 points_dict
                for i in range(len(out)):
                    pred_val = out[i].item()
                    true_val = y_true[i].item()

                    # 如果需要反向对数转换 (针对 perf)
                    if FLAGS.encode_log and target_name == 'actual_perf':
                        pred_val = 2 ** (pred_val) * (1 / FLAGS.normalizer)

                    points_dict[target_name]['pred'].append(pred_val)
                    points_dict[target_name]['true'].append(true_val)

    avg_loss = total_loss / len(loader)
    avg_loss_dict = {key: v / len(loader) for key, v in loss_dict.items()}

    # --- 打印本轮 Loss 详情 (10^10 效果) ---
    print(f"\n>> [Epoch {epoch + 1} {tvt}] Average Loss: {avg_loss:.8e}")
    for k, v in avg_loss_dict.items():
        print(f"   - {k:10s} Loss: {v:.8e}")

    # 绘图逻辑保持原样
    if FLAGS.plot_pred_points and tvt == 'test' and (plot_test or (test_losses and avg_loss < min(test_losses))):
        from src.utils import plot_points, plot_points_with_subplot
        if not FLAGS.multi_target:
            plot_points({f'{FLAGS.target[0]}-pred_points': points_dict[f'{FLAGS.target[0]}']['pred'],
                         f'{FLAGS.target[0]}-true_points': points_dict[f'{FLAGS.target[0]}']['true']},
                        f'epoch_{epoch + 1}_{tvt}', saver.get_log_dir())
        else:
            plot_points_with_subplot(points_dict, f'epoch_{epoch + 1}_{tvt}', saver.get_log_dir(), target_list)

    # 推理评估逻辑
    if FLAGS.subtask == 'inference':
        # from utils import _report_rmse_etc
        _report_rmse_etc(points_dict, f'epoch {epoch}:', True)

    return avg_loss, avg_loss_dict, points_dict


from torch_geometric.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, data_list):
        super(SimpleDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


if __name__ == "__main__":
    # 1. 定义你的 AST 数据集路径 (请确保路径准确)0
    # AST_PATH = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_dataset/graph"    # baseline
    AST_PATH = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_datasets/code/ast_final_dataset_1"
    # AST_PATH = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_dataset/code"


    print(f"🚀 正在从 {AST_PATH} 加载 AST 数据集...")
    ast_data_list = []

    # 2. 递归扫描所有 .pt 文件
    for root, _, files in os.walk(AST_PATH):
        for file in files:
            if file.endswith(".pt"):
                full_path = os.path.join(root, file)
                try:
                    data = torch.load(full_path, map_location='cpu')


                    data.file=file      # 三特征融合加入


                    # print(f"DEBUG: 数据包含的属性有: {data.keys}") km

                    # 补丁：确保包含 edge_attr (PNA 模型必需)
                    # if not hasattr(data, 'edge_attr') or data.edge_attr is None:
                    #     # sys.exists(1)
                    #     num_edges = data.edge_index.shape[1]
                    #     data.edge_attr = torch.zeros((num_edges, 7))


                    ast_data_list.append(data)
                except Exception as e:
                    print(f"⚠️ 跳过损坏文件 {full_path}: {e}")

    print(f"✅ 加载完成，总样本数: {len(ast_data_list)}")

    if len(ast_data_list) == 0:
        print("❌ 错误：未找到任何 .pt 数据文件！请检查路径。")
        exit(1)

    # 3. 封装进 Dataset 容器，不再引用 programl_data
    dataset = SimpleDataset(ast_data_list)
    modes = ["all", "ast_cdfg", "ast_text", "cdfg_text", "cdfg_only", "ast_only", "LM"]
    # 4. 根据 FLAGS 进入训练或推理

    for i in range(1, 5):
        # train_main(dataset, "ast_only", i)
        inference(dataset, "ast_only", i)

