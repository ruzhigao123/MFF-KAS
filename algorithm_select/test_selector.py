
import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import json
import sys
from tqdm import tqdm
import numpy as np

# --- 1. 基础路径与环境配置 ---
BASE_DIR = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master"
sys.path.append(BASE_DIR)

from select_model import AlgorithmSelector
from select_dataset import AlgorithmSelectDataset
from make_labels import get_aligned_data

# --- 2. 常量定义 (严格与训练逻辑互补) ---
MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'doitgen',
               'mvt', 'fdtd-2d', 'gemver', 'gemm-p', 'gesummv',
               'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d']

# 训练时使用的边界 (与你提供的 select_main.py 一致)
TRAIN_MINX, TRAIN_PINX = 5, 10


def get_unseen_test_list(all_aligned_info):
    """
    提取模型从未见过的 Kernel 样本
    """
    train_kernels = set(MACHSUITE_KERNEL[:TRAIN_MINX] + poly_KERNEL[:TRAIN_PINX])
    test_list = []

    print(">>> 正在筛选 Unseen Kernels (测试集)...")
    for info in all_aligned_info:
        k_name = info['key'][1]

        # 如果该 Kernel 不在训练名单中，则加入测试集
        if k_name not in train_kernels:
            # 同样注入路径信息，确保 Dataset 能提取 25 维特征
            data_obj = torch.load(info['cdfg'], map_location='cpu')
            data_obj.file = os.path.basename(info['cdfg'])
            data_obj.kernel = k_name
            data_obj.ast_path = info.get('ast', "").replace('.ast.pt', '.ast.json')

            test_list.append(data_obj)

    return test_list


from select_main import extract_hybrid_meta
LABEL_PATH = os.path.join(BASE_DIR, "algorithm_select/labels.json")

# --- 3. 主测试程序 ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = os.path.join(BASE_DIR, "save_models_and_data/best_selector_nn.pth")

    # 1. 获取所有对齐数据
    all_aligned = get_aligned_data()[0]

    # 2. 获取测试样本 (从未见过的 Kernel)
    test_samples = get_unseen_test_list(all_aligned)
    print(f"✅ 找到来自测试 Kernel 的样本数量: {len(test_samples)}")

    # 3. 封装 Dataset 和 DataLoader
    # 确保此处提取的是 25 维特征

    with open(LABEL_PATH, 'r') as f:
        label_dict = {eval(k): v for k, v in json.load(f).items()}


    test_set = AlgorithmSelectDataset(
        test_samples,  # 第一个参数：数据列表
        label_dict,  # 第二个参数：标签字典 (传空字典即可)
        extract_hybrid_meta  # 第三个参数：特征提取函数
    )
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # 4. 初始化模型 (参数需与 select_model.py 的最新修改一致)
    # 注意：如果你修改了 meta_processor，这里的维度会自动对齐
    model = AlgorithmSelector(
        gcn_params={'hidden_channels': 128},
        meta_base_dim=9,
        ast_tail_dim=16,
        num_tools=7
    ).to(device)

    # 5. 加载权重
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f">>> 成功加载模型权重: {MODEL_PATH}")
    else:
        print(f"❌ 错误: 找不到权重文件 {MODEL_PATH}")
        sys.exit()

    model.eval()

    # 6. 执行推理测试
    total_correct = 0
    # 统计各个指标的分类情况
    target_names = ['PERF', 'LUT', 'FF', 'DSP', 'BRAM']
    target_stats = {i: {'correct': 0, 'total': 0} for i in range(5)}

    print(f"\n>>> 正在对测试集进行分类评估...")
    with torch.no_grad():
        for data, meta, m_id, label in tqdm(test_loader):
            data, meta, m_id, label = data.to(device), meta.to(device), m_id.to(device), label.to(device)

            logits = model(data, meta, m_id)
            preds = logits.argmax(dim=1)

            total_correct += (preds == label).sum().item()

            for i in range(5):
                mask = (m_id == i)
                if mask.any():
                    target_stats[i]['correct'] += (preds[mask] == label[mask]).sum().item()
                    target_stats[i]['total'] += mask.sum().item()

    # 7. 输出详细报告
    overall_acc = total_correct / len(test_set) if len(test_set) > 0 else 0

    print("\n" + "=" * 60)
    print("🚀 推荐系统 (Selector) Unseen Kernels 分类报告")
    print("=" * 60)
    print(f"{'指标类型':<12} | {'准确率 (Acc)':<15} | {'样本总数':<10}")
    print("-" * 60)

    for i in range(5):
        stats = target_stats[i]
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{target_names[i]:<12} | {acc:^15.2%} | {stats['total']:^10}")

    print("-" * 60)
    print(f"{'总体平均 (Overall)':<12} | {overall_acc:^15.2%} | {len(test_set):^10}")
    print("=" * 60)