# fold_config.py

# 你的所有 Kernel 列表 (共 22 个)
ALL_KERNEL = [
    'aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw',
    '2mm', '3mm', 'adi', 'atax', 'bicg', 'doitgen', 'mvt', 'fdtd-2d', 'gemver',
    'gemm-p', 'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d'
]

# 将 22 个 Kernel 均匀分配到 5 个 Fold 中 (每组 4-5 个)
# 分配逻辑：手动交叉排列以保证分布均匀
K_FOLDS = [
    ['aes', 'gemm-p', '2mm', 'bicg', 'heat-3d'],  # Fold 0
    ['gemm-blocked', 'stencil', '3mm', 'doitgen', 'jacobi-1d'],  # Fold 1
    ['gemm-ncubed', 'nw', 'adi', 'mvt', 'jacobi-2d'],  # Fold 2
    ['spmv-crs', 'atax', 'fdtd-2d', 'seidel-2d'],  # Fold 3
    ['gemver', 'spmv-ellpack', 'gesummv']  # Fold 4'spmv-ellpack'
]


def get_fold_idx(kernel_name):
    """
    根据 Kernel 名称返回它所属的 Fold 编号 (0-4)
    """
    for i, fold in enumerate(K_FOLDS):
        if kernel_name in fold:
            return i
    # 如果找不到，默认归为 Fold 0 (防御性编程)
    return 0


if __name__ == "__main__":
    # 测试代码：验证是否所有 Kernel 都被分配了
    assigned = []
    for f in K_FOLDS:
        assigned.extend(f)

    print(f"原始 Kernel 总数: {len(ALL_KERNEL)}")
    print(f"已分配 Kernel 总数: {len(assigned)}")

    missing = set(ALL_KERNEL) - set(assigned)
    if missing:
        print(f"⚠️ 警告！以下 Kernel 未被分配: {missing}")
    else:
        print("✅ 所有 Kernel 已成功分配到 5-Fold 中。")