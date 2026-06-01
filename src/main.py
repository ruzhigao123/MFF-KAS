
import time
import os
import torch
import os.path as osp
from os.path import join
from tqdm import tqdm

from config import FLAGS
from train import train_main, inference
from LLM4DSE.dse import LMEAExplorer, EAExplorer, SAExplorer, ExhaustiveExplorer, ACOExplorer, LMSAExplorer, \
    LMACOExplorer
from saver import saver
from src.get_root_path import get_root_path
import config
from programl_data import get_data_list
from dse_database.gen_dataset import MyOwnDataset

TARGETS = config.TARGETS
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL
poly_KERNEL = config.poly_KERNEL

import os
import torch
from torch_geometric.data import Dataset
from config import FLAGS
from train_del_0 import train_main, inference
import sys


# 定义一个极简的容器类，用来存放我们手动加载的 data_list
class SimpleDataset(Dataset):
    def __init__(self, data_list):
        super(SimpleDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


if __name__ == '__main__':
    # 1. 定义你的 AST 数据集路径 (请确保路径准确)0
    AST_PATH = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_dataset/graph"    # baseline
    # AST_PATH = "/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_datasets/code/ast_final_dataset_1"
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
                    ast_data_list.append(data)
                except Exception as e:
                    print(f"⚠️ 跳过损坏文件 {full_path}: {e}")

    print(f"✅ 加载完成，总样本数: {len(ast_data_list)}")

    if len(ast_data_list) == 0:
        print("❌ 错误：未找到任何 .pt 数据文件！请检查路径。")
        exit(1)

    # 3. 封装进 Dataset 容器，不再引用 programl_data
    dataset = SimpleDataset(ast_data_list)

    _data_dir = join(get_root_path(), 'data')
    print(f"[main] FLAGS.subtask={FLAGS.subtask!r}")
    print(f"[main] 若为 inference：预测 CSV 写入目录 {os.path.abspath(_data_dir)}（项目根下 data/）")

    # 4. 根据 FLAGS 进入训练或推理
    if FLAGS.subtask == 'train':
        train_main(dataset)
    elif FLAGS.subtask == 'inference':
        inference(dataset)
