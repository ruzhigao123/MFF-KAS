"""
Causal-aware data utilities
用于构造 intervention-style 训练样本对的工具函数
"""
from typing import Dict, List, Tuple, Optional
from src.parameter import DesignPoint
import random
import numpy as np


def parse_key_to_design_point(key: str) -> DesignPoint:
    """
    从 key 字符串解析出 DesignPoint
    
    Args:
        key: 格式如 "lv2.__PIPE__L1-flatten.__TILE__L2-4"
    
    Returns:
        DesignPoint: Dict[str, Union[int, str]]
    """
    # 移除 'lv2.' 前缀（如果存在）
    if key.startswith('lv2.'):
        key = key[4:]
    elif key.startswith('lv1.'):
        key = key[4:]
    
    point = {}
    if not key:
        return point
    
    # 按 '.' 分割
    parts = key.split('.')
    for part in parts:
        if '-' in part:
            # 格式: "__PIPE__L1-flatten"
            pragma_id, value = part.rsplit('-', 1)
            # 尝试将 value 转换为 int，如果失败则保持为 str
            try:
                value = int(value)
            except ValueError:
                # 检查是否是 'NA'
                if value == 'NA':
                    value = ''
                # 否则保持为字符串
                pass
            point[pragma_id] = value
    
    return point


def hamming_distance(point1: DesignPoint, point2: DesignPoint) -> int:
    """
    计算两个 DesignPoint 之间的 Hamming 距离（不同 pragma 的数量）
    """
    all_keys = set(point1.keys()) | set(point2.keys())
    distance = 0
    for key in all_keys:
        val1 = point1.get(key, None)
        val2 = point2.get(key, None)
        if val1 != val2:
            distance += 1
    return distance


def create_intervention_pairs(
    points: List[DesignPoint],
    max_distance: int = 2,
    num_pairs: Optional[int] = None,
    seed: int = 42
) -> List[Tuple[DesignPoint, DesignPoint]]:
    """
    从一组 DesignPoint 中创建 intervention-style 样本对
    
    Args:
        points: List of DesignPoint
        max_distance: 最大 Hamming 距离（默认2，即最多2个 pragma 不同）
        num_pairs: 要生成的样本对数量（None 表示生成所有可能的对）
        seed: 随机种子
    
    Returns:
        List of (P, P') tuples
    """
    random.seed(seed)
    np.random.seed(seed)
    
    pairs = []
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i >= j:
                continue
            dist = hamming_distance(p1, p2)
            if 1 <= dist <= max_distance:
                pairs.append((p1, p2))
    
    # 如果指定了数量，随机采样
    if num_pairs is not None and len(pairs) > num_pairs:
        pairs = random.sample(pairs, num_pairs)
    
    return pairs


def create_pairs_from_keys(
    keys: List[str],
    max_distance: int = 2,
    num_pairs: Optional[int] = None,
    seed: int = 42
) -> List[Tuple[str, str, DesignPoint, DesignPoint]]:
    """
    从 key 列表创建样本对（返回 key 和对应的 DesignPoint）
    
    Args:
        keys: List of key strings
        max_distance: 最大 Hamming 距离
        num_pairs: 要生成的样本对数量
        seed: 随机种子
    
    Returns:
        List of (key1, key2, point1, point2) tuples
    """
    # 解析所有 key 为 DesignPoint
    points_dict = {}
    for key in keys:
        point = parse_key_to_design_point(key)
        points_dict[key] = point
    
    # 创建配对
    pairs = []
    key_list = list(points_dict.keys())
    for i, key1 in enumerate(key_list):
        for j, key2 in enumerate(key_list):
            if i >= j:
                continue
            point1 = points_dict[key1]
            point2 = points_dict[key2]
            dist = hamming_distance(point1, point2)
            if 1 <= dist <= max_distance:
                pairs.append((key1, key2, point1, point2))
    
    # 如果指定了数量，随机采样
    if num_pairs is not None and len(pairs) > num_pairs:
        random.seed(seed)
        pairs = random.sample(pairs, num_pairs)
    
    return pairs
