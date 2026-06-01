import os
import torch
from tqdm import tqdm
import sys

# 路径配置
CDFG_DIR = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_dataset/graph"
AST_DIR = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_datasets/code/ast_final_dataset_1"
TEXT_DIR = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_dataset/code"


def get_aligned_data_list():
    """
    根据 ID 自动对齐 CDFG, AST 和 Text 数据
    """
    # 提取 ID 映射
    cdfg_files = {f.split('_')[1].split('.')[0]: f for f in os.listdir(CDFG_DIR) if f.startswith('data_')}
    ast_files = {f.split('.')[0]: f for f in os.listdir(AST_DIR) if f.endswith('.ast.pt')}
    text_files = {f.split('.')[0]: f for f in os.listdir(TEXT_DIR) if f.endswith('.pt') and '_' not in f}

    common_ids = set(cdfg_files.keys()) & set(ast_files.keys()) & set(text_files.keys())
    print(f"--- [Utils] 找到对齐样本数: {len(common_ids)} ---")

    aligned_data = []
    for idx in tqdm(sorted(list(common_ids), key=int), desc="对齐数据中"):
        # 加载
        c_data = torch.load(os.path.join(CDFG_DIR, cdfg_files[idx]), map_location='cpu')
        a_data = torch.load(os.path.join(AST_DIR, ast_files[idx]), map_location='cpu')
        t_data = torch.load(os.path.join(TEXT_DIR, text_files[idx]), map_location='cpu')

        # 注入特征 (Selector 和 Net 需要)
        c_data.ast_feat = a_data.x if hasattr(a_data, 'x') else a_data
        c_data.text_feat = t_data.x if hasattr(t_data, 'x') else t_data
        c_data.sample_id = idx

        # 确保 kernel 属性存在
        if not hasattr(c_data, 'kernel'):
            c_data.kernel = "unknown"

        aligned_data.append(c_data)

    return aligned_data


def get_metric_name(metric_id):
    """任务 ID 转名称"""
    mapping = {0: 'perf', 1: 'util-LUT', 2: 'util-FF', 3: 'util-DSP', 4: 'util-BRAM'}
    return mapping.get(metric_id, "unknown")


def get_tool_name(tool_id):
    """工具 ID 转描述"""
    names = [
        "All-Fusion (3-Modal)", "AST+CDFG", "AST+Text",
        "CDFG+Text", "CDFG-Only", "AST-Only", "Text-Only(LM)"
    ]
    return names[tool_id] if tool_id < len(names) else "Unknown Tool"