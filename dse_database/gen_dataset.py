"""
训练数据生成脚本

本脚本的主要功能：
1. 从.gexf图文件（程序图结构）和.db数据库文件（DSE探索结果）中读取数据
2. 对每个kernel的每个配置点进行特征编码（节点特征、边特征）
3. 使用CodeBERT对源代码进行编码，生成代码特征
4. 构建PyTorch Geometric的Data对象，包含图结构、特征和目标值
5. 保存处理后的数据到two_tower_dataset目录，供训练使用

数据流程：
- 输入：.gexf图文件 + .db DSE结果数据库
- 处理：特征编码 + CodeBERT编码
- 输出：two_tower_dataset/graph/ 和 two_tower_dataset/code/ 目录下的.pt文件
"""
import sys
import os

# 強制把專案根目錄加進 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from LLM4DSE.dse import GNNModel
import os.path as osp
from os.path import join, basename

# 添加项目根目录到Python路径，以便导入src模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import src.config as config
from src.config import FLAGS
from glob import glob, iglob
from src.utils import get_root_path, MLP, print_stats, get_save_path, \
    create_dir_if_not_exists, plot_dist, load
from collections import Counter, OrderedDict
import pickle
from sklearn.preprocessing import OneHotEncoder
import warnings
from tqdm import tqdm
from src.saver import saver
import networkx as nx
from src.programl_data import _encode_X_dict, _encode_edge_dict, _encode_X_torch, _encode_edge_torch, \
    finte_diff_as_quality, create_edge_index
import math
import torch
import numpy as np
from shutil import rmtree
from torch_geometric.data import Data
from copy import deepcopy
from torch_geometric.data import Dataset
# transformers 将在需要时延迟导入，以避免在导入时检查 PyTorch
import pickle
from pathlib import Path

# 数据标签标识
tag = 'new_speedup'

# 从配置文件读取目标变量和kernel列表
TARGETS = config.TARGETS  # 训练目标，如perf, quality, util-BRAM等
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL  # MachSuite基准测试的kernel列表
poly_KERNEL = config.poly_KERNEL  # PolyBench基准测试的kernel列表

# 合并所有kernel
ALL_KERNEL = MACHSUITE_KERNEL + poly_KERNEL

# 重新定义kernel列表（如果config中的定义不完整，这里作为备用）
MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'doitgen',
               'mvt', 'fdtd-2d', 'gemver', 'gemm-p', 'gesummv',
               'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d']

# ==================== 路径配置 ====================
# 代码特征保存路径（CodeBERT编码后的特征）
code_path = join(get_root_path(), 'two_tower_dataset/code')
# 图数据保存路径（PyTorch Geometric Data对象）
graph_path = join(get_root_path(), 'two_tower_dataset/graph')
# points_list.pkl 保存路径（所有 kernel 的 design_point 列表）
points_path = join(get_root_path(), 'two_tower_dataset/points')
# 编码器保存路径（OneHotEncoder等）
ENCODER_PATH = join(get_root_path(), 'passion/save_models_and_data')

# 构建数据库文件路径列表（包含所有benchmark的数据库文件）
db_path = []
for benchmark in FLAGS.benchmarks:
    db_path.append(f'../dse_database/{benchmark}/databases/**/*')

# 程序图文件（.gexf格式）的搜索路径
GEXF_FOLDER = join(get_root_path(), 'dse_database', 'programl', '**', 'processed', '**')

# 获取所有.gexf图文件并排序
GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf')])

if __name__ == '__main__':
    # ==================== 初始化 ====================
    # 使用Python字典替代Redis（用于临时存储和查询.db文件中的数据）
    # 注意：原代码使用Redis，但为了简化部署，改用内存字典
    database = {}  # 使用字典存储数据，key为配置点名称，value为pickle序列化的对象
    print(graph_path)
    bench = ['machsuite', 'poly']
    saver.log_info(f'Found {len(GEXF_FILES)} gexf files under {GEXF_FOLDER}')

    # ==================== 特征统计计数器 ====================
    # 用于统计所有出现的特征类型，以便后续进行OneHot编码
    ntypes = Counter()  # 节点类型统计
    ptypes = Counter()  # pragma类型统计
    numerics = Counter()  # 数值特征统计
    itypes = Counter()  # 整数类型统计
    ftypes = Counter()  # 浮点类型统计
    btypes = Counter()  # 布尔类型统计
    ptypes_edge = Counter()  # 边的pragma类型统计
    ftypes_edge = Counter()  # 边的浮点类型统计

    # ==================== 加载或创建编码器 ====================
    # 如果提供了预训练的编码器路径，则加载；否则创建新的OneHot编码器
    encoders_loaded = False
    if FLAGS.encoder_path != None:
        try:
            # 加载已保存的编码器（用于增量训练或使用已有编码）
            # 注意：不同 sklearn 版本之间反序列化 OneHotEncoder 可能会产生 InconsistentVersionWarning。
            # 一般只影响"复用旧 encoder"场景；如果你想彻底消除风险/警告，直接把 FLAGS.encoder_path 设为 None 重新生成即可。
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                encoders = load(FLAGS.encoder_path)


            # 修复 sklearn 版本兼容性问题：旧版本保存的编码器可能缺少新版本的属性
            def _fix_encoder_compat(encoder):
                """修复旧版本 OneHotEncoder 的兼容性问题"""
                # sklearn 1.0+ 新增的私有属性，旧版本没有，可以安全设置
                if not hasattr(encoder, '_infrequent_enabled'):
                    try:
                        encoder._infrequent_enabled = False
                    except:
                        pass
                # 对于只读属性，使用 object.__setattr__ 绕过属性检查
                if not hasattr(encoder, '_infrequent_indices'):
                    try:
                        object.__setattr__(encoder, '_infrequent_indices', None)
                    except:
                        pass  # 如果设置失败，忽略
                # infrequent_categories_ 是只读属性，尝试通过 __dict__ 设置（如果失败就忽略）
                if not hasattr(encoder, 'infrequent_categories_'):
                    try:
                        # 使用 object.__setattr__ 绕过只读属性检查
                        object.__setattr__(encoder, 'infrequent_categories_', [])
                    except:
                        # 如果还是失败，尝试通过 __dict__ 直接设置
                        try:
                            encoder.__dict__['infrequent_categories_'] = []
                        except:
                            pass  # 如果都失败，忽略（可能在使用时会有问题，但至少不会立即崩溃）
                return encoder


            enc_ntype = _fix_encoder_compat(encoders['enc_ntype'])
            enc_ptype = _fix_encoder_compat(encoders['enc_ptype'])
            enc_itype = _fix_encoder_compat(encoders['enc_itype'])
            enc_ftype = _fix_encoder_compat(encoders['enc_ftype'])
            enc_btype = _fix_encoder_compat(encoders['enc_btype'])

            enc_ftype_edge = _fix_encoder_compat(encoders['enc_ftype_edge'])
            enc_ptype_edge = _fix_encoder_compat(encoders['enc_ptype_edge'])
            encoders_loaded = True
            saver.log_info(f"Successfully loaded encoders from {FLAGS.encoder_path}")
        except Exception as e:
            # 如果加载或修复编码器失败，回退到创建新编码器
            saver.warning(f"Failed to load/fix encoders from {FLAGS.encoder_path}: {e}")
            saver.warning("Falling back to creating new encoders...")
            encoders_loaded = False

    if not encoders_loaded:
        # 创建新的OneHot编码器
        # handle_unknown='ignore' 对于处理新kernel中未知的变量类型至关重要
        enc_ntype = OneHotEncoder(handle_unknown='ignore')
        enc_ptype = OneHotEncoder(handle_unknown='ignore')
        enc_itype = OneHotEncoder(handle_unknown='ignore')
        enc_ftype = OneHotEncoder(handle_unknown='ignore')
        enc_btype = OneHotEncoder(handle_unknown='ignore')

        enc_ftype_edge = OneHotEncoder(handle_unknown='ignore')
        enc_ptype_edge = OneHotEncoder(handle_unknown='ignore')
        saver.log_info("Created new OneHot encoders")

    # ==================== 数据存储变量 ====================
    data_list = []  # 存储处理后的Data对象列表
    all_gs = OrderedDict()  # 存储所有图对象，key为kernel名称

    # 收集所有特征用于编码器拟合
    X_ntype_all = []  # 所有节点的类型特征
    X_ptype_all = []  # 所有节点的pragma类型特征
    X_itype_all = []  # 所有节点的整数类型特征
    X_ftype_all = []  # 所有节点的浮点类型特征
    X_btype_all = []  # 所有节点的布尔类型特征

    edge_ftype_all = []  # 所有边的浮点类型特征
    edge_ptype_all = []  # 所有边的pragma类型特征

    tot_configs = 0  # 总配置数量统计
    num_files = 0  # 处理的文件数量统计
    init_feat_dict = {}  # 初始特征字典

    # 测试用的特定文件列表（已注释）
    # gf = ['/home/wslcccc/passion/dse_database/programl/poly/processed/atax_processed_result.gexf', '/home/wslcccc/passion/dse_database/programl/poly/processed/heat-3d_processed_result.gexf']
    # for gexf_file in tqdm(gf):

    # ==================== 主循环：处理每个.gexf图文件 ====================
    skipped_kernels = []  # 记录被跳过的kernel
    for gexf_file in tqdm(GEXF_FILES[0:]):
        # ==================== 过滤：只处理指定kernel的图文件 ====================
        if FLAGS.dataset == 'machsuite' or 'programl' in FLAGS.dataset:
            proceed = False
            matched_kernel = None
            # 检查当前文件是否属于我们要处理的kernel列表
            for k in ALL_KERNEL:
                if k in gexf_file:
                    proceed = True
                    matched_kernel = k
                    break
            if not proceed:
                skipped_kernels.append(gexf_file)
                saver.log_info(f'[SKIP] Skipped gexf file (not in ALL_KERNEL): {basename(gexf_file)}')
                continue  # 跳过不在列表中的kernel
            # pass
        else:
            raise NotImplementedError()

        # ==================== 读取图文件 ====================
        # 使用NetworkX读取.gexf格式的程序图
        g = nx.read_gexf(gexf_file)
        # 初始化variants字典，用于存储该kernel的所有配置变体
        g.variants = OrderedDict()
        # 提取文件名（不含扩展名）
        gname = basename(gexf_file).split('.')[0]  # 完整文件名，如"atax_processed_result"
        saver.log_info(gname)
        n = basename(gexf_file).split('_')[0]  # kernel名称，如"atax"
        all_gs[n] = g  # 保存图对象

        # ==================== 查找对应的数据库文件 ====================
        # 获取存储该kernel的DSE探索结果的.db文件路径
        if FLAGS.dataset == 'programl':
            db_paths = []
            # 在所有数据库路径中搜索包含当前kernel名称的.db文件
            saver.log_info(f'Searching for .db files for kernel: {n}')
            for db_p in db_path:
                saver.log_info(f'  Searching in pattern: {db_p}')
                paths = [f for f in iglob(db_p, recursive=True) if f.endswith('.db') and n in f]
                saver.log_info(f'  Found {len(paths)} matching files in this pattern')
                db_paths.extend(paths)

            # 修复：检查空列表而不是None（db_paths永远不会是None，只会是空列表）
            if len(db_paths) == 0:
                saver.warning(f'No database found for {n} (kernel name: "{n}"). Skipping this kernel.')
                saver.warning(f'  Searched in patterns: {db_path}')
                saver.warning(f'  All .gexf files found: {len(GEXF_FILES)}')
                saver.warning(f'  Current gexf file: {gexf_file}')
                continue
        else:
            raise NotImplementedError()

        # ==================== 加载数据库到内存字典 ====================
        # 清空字典，准备加载新的数据
        database.clear()
        saver.log_info(f'db_paths for {n}: ({len(db_paths)} files found)')
        for d in db_paths:
            saver.log_info(f'  {d}')

        assert len(db_paths) >= 1  # 确保至少有一个数据库文件

        # 加载数据库文件并获取所有键值对
        # 每个键（key）表示一个配置点，值（value）是该配置点的DSE结果（性能、资源使用等）
        # key的格式显示了源代码中每个pragma的值
        for idx, file in enumerate(db_paths):
            with open(file, 'rb') as f_db:
                data = pickle.load(f_db)  # 加载pickle格式的数据库
            if not isinstance(data, dict):
                # 这个脚本后续逻辑依赖 “key->value” 形式
                raise ValueError(f'Unexpected data format in {file}: {type(data)} (expected dict)')

            # 关键修复：统一把 key 规范化为 str，避免 bytes/str 混用导致 KeyError
            for k, v in data.items():
                if isinstance(k, bytes):
                    k_str = k.decode('utf-8')
                else:
                    k_str = str(k)
                database[k_str] = v

            max_idx = idx + 1

        # ==================== 获取所有配置键 ====================
        # 此时 database 的 key 已全部是 str
        keys = list(database.keys())
        lv2_keys = [k for k in keys if 'lv2' in k]  # 过滤出lv2级别的键
        saver.log_info(f'num keys for {n}: {len(keys)} and lv2 keys: {len(lv2_keys)}')

        # ==================== 查找参考点（性能最优的配置） ====================
        # 参考点用于计算quality指标（相对于最优性能的差异）
        got_reference = False
        res_reference = 0
        max_perf = 0
        # 遍历所有配置点，找到性能最优的作为参考点
        for key in sorted(keys):
            pickle_obj = database[key]  # 从字典获取值
            # 如果值是bytes，需要先处理；如果已经是对象，直接使用
            if isinstance(pickle_obj, bytes):
                # 替换pickle中的模块路径（兼容性处理）
                obj = pickle.loads(pickle_obj.replace(b'localdse', b'autodse'))
            else:
                # 如果已经是反序列化的对象，直接使用
                obj = pickle_obj

            if type(obj) is int or type(obj) is dict:
                continue  # 跳过无效数据
            if key[0:3] == 'lv1':
                continue  # 跳过lv1级别的配置
            # 更新最大性能值
            if obj.perf > max_perf:
                max_perf = obj.perf
                got_reference = True
                res_reference = obj
        if res_reference != 0:
            saver.log_info(f'reference point for {n} is {res_reference.perf}')
        else:
            saver.log_info(f'did not find reference point for {n} with {len(keys)} points')

        # ==================== 处理每个配置点 ====================
        # 遍历所有配置点，为每个配置生成特征编码
        for key in sorted(keys):
            pickle_obj = database[key]  # 从字典获取值
            # 如果值是bytes，需要先处理；如果已经是对象，直接使用
            if isinstance(pickle_obj, bytes):
                obj = pickle.loads(pickle_obj.replace(b'localdse', b'autodse'))
            else:
                # 如果已经是反序列化的对象，直接使用
                obj = pickle_obj
            # try:
            if type(obj) is int or type(obj) is dict:
                continue  # 跳过无效数据
            # 回归任务中跳过lv1级别的配置（通常质量较低）
            if FLAGS.task == 'regression' and key[0:3] == 'lv1':  # or obj.perf == 0:#obj.ret_code.name == 'PASS':
                continue
            # 可选：跳过性能为0的配置（已注释）
            # if FLAGS.task == 'regression' and not FLAGS.invalid and obj.perf == 0:
            #     continue
            print(obj.point)  # 打印配置点信息

            # ==================== 特征编码 ====================
            # 对节点特征进行编码（包括节点类型、pragma类型、数值特征等）
            xy_dict = _encode_X_dict(
                g, ntypes=ntypes, ptypes=ptypes, itypes=itypes, ftypes=ftypes, btypes=btypes, numerics=numerics,
                obj=obj)
            # 对边特征进行编码
            edge_dict = _encode_edge_dict(
                g, ftypes=ftypes_edge, ptypes=ptypes_edge)

            # ==================== 目标值计算 ====================
            # 根据任务类型计算目标值（回归任务）
            if FLAGS.task == 'regression':
                for tname in TARGETS:
                    if tname == 'perf':
                        # 性能目标：使用对数归一化
                        if obj.perf > 0:
                            y = FLAGS.normalizer / obj.perf  # 归一化（性能越高，y越大）
                            y = math.log(y + 1, 100)  # 对数变换
                        else:
                            y = obj.perf
                        # 保存实际性能值（未归一化）
                        xy_dict['actual_perf'] = torch.FloatTensor(np.array([obj.perf]))
                    elif tname == 'quality':
                        # 质量目标：相对于参考点的有限差分
                        y = finte_diff_as_quality(obj, res_reference)
                    elif 'util' in tname or 'total' in tname:
                        # 资源利用率目标：对数变换
                        y = math.log(obj.res_util[tname] + 1, 100)
                    else:
                        raise NotImplementedError()
                    xy_dict[tname] = torch.FloatTensor(np.array([y]))
            else:
                raise NotImplementedError()

            # ==================== 保存变体数据 ====================
            vname = key  # 使用key作为变体名称
            # 将编码后的节点特征、边特征和设计点保存到图的variants字典中
            # 保存 point 用于因果模型训练
            g.variants[vname] = (xy_dict, edge_dict, obj.point)

            # ==================== 收集特征用于编码器拟合 ====================
            # 将所有特征添加到全局列表中，后续用于拟合OneHot编码器
            X_ntype_all += xy_dict['X_ntype']
            X_ptype_all += xy_dict['X_ptype']
            X_itype_all += xy_dict['X_itype']
            X_ftype_all += xy_dict['X_ftype']
            X_btype_all += xy_dict['X_btype']

            edge_ftype_all += edge_dict['X_ftype']
            edge_ptype_all += edge_dict['X_ptype']

        # ==================== 统计信息 ====================
        tot_configs += len(g.variants)  # 累计配置总数
        num_files += 1  # 累计文件数
        saver.log_info(f'{n} g.variants {len(g.variants)} tot_configs {tot_configs}')
        saver.log_info(f'\tntypes {len(ntypes)}')
        saver.log_info(f'\tptypes {len(ptypes)} {ptypes}')
        saver.log_info(f'\tnumerics {len(numerics)} {numerics}')

    # ==================== 处理完成统计 ====================
    saver.log_info(f'\n{"=" * 60}')
    saver.log_info(f'[SUMMARY] Processing Summary:')
    saver.log_info(f'  Total .gexf files found: {len(GEXF_FILES)}')
    saver.log_info(f'  Kernels successfully processed: {num_files}')
    saver.log_info(f'  Kernels skipped (not in ALL_KERNEL): {len(skipped_kernels)}')
    if skipped_kernels:
        saver.log_info(f'  Skipped files:')
        for sf in skipped_kernels:
            saver.log_info(f'    - {basename(sf)}')
    saver.log_info(f'  Expected kernels (MACHSUITE): {len(MACHSUITE_KERNEL)} - {MACHSUITE_KERNEL}')
    saver.log_info(f'  Expected kernels (POLY): {len(poly_KERNEL)} - {poly_KERNEL}')
    saver.log_info(f'  Total expected: {len(ALL_KERNEL)}')
    saver.log_info(f'{"=" * 60}\n')

    # ==================== 拟合编码器（在所有数据收集完成后） ====================
    # 如果创建了新编码器，需要先拟合才能使用
    if not encoders_loaded:
        saver.log_info("Fitting OneHot encoders with collected features...")
        # 将列表转换为numpy数组用于fit（需要是2D数组）
        if len(X_ntype_all) > 0:
            enc_ntype.fit(np.array(X_ntype_all).reshape(-1, 1))
        if len(X_ptype_all) > 0:
            enc_ptype.fit(np.array(X_ptype_all).reshape(-1, 1))
        if len(X_itype_all) > 0:
            enc_itype.fit(np.array(X_itype_all).reshape(-1, 1))
        if len(X_ftype_all) > 0:
            enc_ftype.fit(np.array(X_ftype_all).reshape(-1, 1))
        if len(X_btype_all) > 0:
            enc_btype.fit(np.array(X_btype_all).reshape(-1, 1))
        if len(edge_ftype_all) > 0:
            enc_ftype_edge.fit(np.array(edge_ftype_all).reshape(-1, 1))
        if len(edge_ptype_all) > 0:
            enc_ptype_edge.fit(np.array(edge_ptype_all).reshape(-1, 1))
        saver.log_info("OneHot encoders fitted successfully")

        # ==================== 加载CodeBERT模型 ====================
        # 用于对源代码进行编码，生成代码特征向量
        # 说明：
        # - from_pretrained(path) 要求该目录下存在权重文件：pytorch_model.bin 或 model.safetensors 等
        # - 你当前的 codebert/ 目录是空的，只看到 .cache/huggingface/download 里的 .lock/.metadata，说明模型并未完整下载
        # 解决方式：
        # - 如果你有网络：用 HuggingFace 模型名自动下载，并把 cache_dir 指到本项目 codebert/，后续可离线复用
        # - 如果你离线：需要把真正的权重文件放到某个目录，再把 CODEBERT_DIR 指向那个目录

        # 首先验证 PyTorch 是否可用
        try:
            saver.log_info(f"PyTorch version: {torch.__version__}")
            # 测试创建张量
            test_tensor = torch.tensor([1.0])
            saver.log_info("PyTorch tensor creation test passed:", test_tensor)
            # 检查 CUDA
            if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'is_available'):
                cuda_available = torch.cuda.is_available()
                saver.log_info(f"CUDA available: {cuda_available}")
            else:
                saver.log_info("CUDA check not available")
        except Exception as e:
            saver.error(f"PyTorch test failed: {e}")
            if 'torch' in sys.modules:
                saver.error(f"PyTorch module: {sys.modules['torch']}")
            raise

        # 在调用 from_pretrained 之前，再次确保 PyTorch 可用
        # 这可以解决 transformers 在运行时检查 PyTorch 的问题
        import importlib

        # 确保 torch 在 sys.modules 中并且是最新的
        if 'torch' not in sys.modules:
            import torch

            saver.log_info("Re-imported torch before loading model")
        else:
            # 重新加载 torch 模块以确保 transformers 能检测到
            # torch = importlib.reload(sys.modules['torch'])
            saver.log_info(f"Reloaded torch module: {torch.__version__}")

        # 验证 torch 是否真的可用
        # try:
        #     _ = torch.tensor([1.0])
        #     saver.log_info("PyTorch verification before model loading: OK")
        # except Exception as e:
        #     saver.error(f"PyTorch verification failed before model loading: {e}")
        #     raise

        # 为 torch 添加 compiler 补丁（transformers 4.57.6 需要 torch.compiler，但 PyTorch 2.0.1 没有）
        # 重要：必须在导入 transformers 之前修补！
        try:
            if not hasattr(torch, 'compiler'):
                # 添加一个兼容的 compiler 模块
                class _CompilerModule:
                    """兼容性补丁：为旧版 PyTorch 添加 torch.compiler"""

                    @staticmethod
                    def disable(recursive=False):
                        """空实现，因为旧版 PyTorch 不支持编译器"""

                        def decorator(func):
                            return func

                        return decorator


                torch.compiler = _CompilerModule()
                saver.log_info("✓ Patched torch.compiler for compatibility")
            else:
                saver.log_info("✓ torch.compiler already exists")
        except Exception as e:
            saver.warning(f"Could not patch torch.compiler: {e}")
            # 继续执行，可能不需要这个补丁

        # 为 torch 添加 float8 dtype 补丁（transformers 4.57.6 在 tensor_parallel 中引用）
        # PyTorch 2.0.1 不包含 float8_* 系列
        try:
            if not hasattr(torch, 'float8_e4m3fn'):
                torch.float8_e4m3fn = torch.float16
                saver.log_info("✓ Patched torch.float8_e4m3fn for compatibility")
            if not hasattr(torch, 'float8_e4m3fnuz'):
                torch.float8_e4m3fnuz = torch.float16
                saver.log_info("✓ Patched torch.float8_e4m3fnuz for compatibility")
            if not hasattr(torch, 'float8_e5m2'):
                torch.float8_e5m2 = torch.float16
                saver.log_info("✓ Patched torch.float8_e5m2 for compatibility")
            if not hasattr(torch, 'float8_e5m2fnuz'):
                torch.float8_e5m2fnuz = torch.float16
                saver.log_info("✓ Patched torch.float8_e5m2fnuz for compatibility")
        except Exception as e:
            saver.warning(f"Could not patch torch float8 dtypes: {e}")

        # 为 torch 添加 _dynamo 补丁（transformers 4.57.6 使用 torch._dynamo.allow_in_graph）
        # PyTorch 2.0.1 可能没有 torch._dynamo
        try:
            if not hasattr(torch, '_dynamo'):
                class _DynamoModule:
                    """兼容性补丁：为旧版 PyTorch 添加 torch._dynamo"""

                    @staticmethod
                    def allow_in_graph(fn=None, **kwargs):
                        if fn is None:
                            def decorator(func):
                                return func

                            return decorator
                        return fn


                torch._dynamo = _DynamoModule()
                saver.log_info("✓ Patched torch._dynamo for compatibility")
            else:
                saver.log_info("✓ torch._dynamo already exists")
        except Exception as e:
            saver.warning(f"Could not patch torch._dynamo: {e}")

        # 兼容 torch.contiguous_format（某些环境下 _refs.empty 不认识这个 memory_format）
        try:
            if hasattr(torch, "contiguous_format"):
                if torch.contiguous_format is not None:
                    torch.contiguous_format = None
                    saver.log_info("✓ Patched torch.contiguous_format to None for compatibility")
        except Exception as e:
            saver.warning(f"Could not patch torch.contiguous_format: {e}")

        # 为 torch.empty 添加 memory_format 兼容补丁
        # 某些环境下 torch._refs.empty 会报 unknown memory format torch.contiguous_format
        try:
            _orig_empty = torch.empty


            def _strip_memory_format_args(args, kwargs):
                # memory_format 可能以关键字或位置参数传入
                if "memory_format" in kwargs:
                    kwargs.pop("memory_format", None)
                else:
                    # torch.empty(size, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None)
                    if len(args) >= 6:
                        args = list(args)
                        # 将 memory_format 置为 None
                        args[5] = None
                        args = tuple(args)
                return args, kwargs


            def _empty_compat(*args, **kwargs):
                # 无条件覆盖 memory_format，避免默认值携带 torch.contiguous_format
                args, kwargs = _strip_memory_format_args(args, kwargs)
                kwargs["memory_format"] = None
                return _orig_empty(*args, **kwargs)


            torch.empty = _empty_compat
            saver.log_info("✓ Patched torch.empty for memory_format compatibility")
        except Exception as e:
            saver.warning(f"Could not patch torch.empty: {e}")

        # 同步修补 torch._refs.empty（transformers 在加载权重时会走这里）
        try:
            import torch._refs as torch_refs

            if hasattr(torch_refs, "empty"):
                _orig_refs_empty = torch_refs.empty


                def _refs_empty_compat(*args, **kwargs):
                    args, kwargs = _strip_memory_format_args(args, kwargs)
                    kwargs["memory_format"] = None
                    return _orig_refs_empty(*args, **kwargs)


                torch_refs.empty = _refs_empty_compat
                saver.log_info("✓ Patched torch._refs.empty for memory_format compatibility")
        except Exception as e:
            saver.warning(f"Could not patch torch._refs.empty: {e}")

        # 尝试设置环境变量，确保 transformers 能检测到 PyTorch
        # 某些版本的 transformers 可能依赖这些环境变量
        # 重要：必须在导入 transformers 之前设置！
        os.environ['TORCH_AVAILABLE'] = '1'
        if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
            os.environ['CUDA_AVAILABLE'] = '1'

        # 在导入 transformers 之前，修补 PyTorch 的 _pytree 模块
        # transformers 4.57.6 需要 register_pytree_node，但 PyTorch 2.0.1 可能没有
        # 重要：必须在导入 transformers 之前修补！
        try:
            import torch.utils._pytree as pytree_module

            saver.log_info(f"Checking torch.utils._pytree for register_pytree_node...")
            if not hasattr(pytree_module, 'register_pytree_node'):
                # 存储注册的函数，供 transformers 使用
                _registered_pytree_nodes = {}


                def register_pytree_node(cls, flatten_fn, unflatten_fn, **kwargs):
                    """兼容性补丁：为旧版 PyTorch 添加 register_pytree_node"""
                    # 接受额外的关键字参数（如 serialized_type_name），但忽略它们
                    # 因为旧版 PyTorch 不需要这些参数
                    # 存储注册的函数
                    _registered_pytree_nodes[cls] = (flatten_fn, unflatten_fn)
                    # 将函数存储到模块中，供 transformers.utils.generic 使用
                    if flatten_fn is not None:
                        pytree_module.__dict__['_model_output_flatten'] = flatten_fn
                    if unflatten_fn is not None:
                        pytree_module.__dict__['_model_output_unflatten'] = unflatten_fn


                pytree_module.register_pytree_node = register_pytree_node
                pytree_module._registered_pytree_nodes = _registered_pytree_nodes


                # 预先定义这些变量，即使还没有注册
                # transformers 会在导入时注册，但 generic.py 在导入时就需要这些变量
                # 所以我们提供一个默认实现
                def _default_flatten(output):
                    if hasattr(output, '__dict__'):
                        values = []
                        for key, value in output.__dict__.items():
                            if not key.startswith('_'):
                                values.append(value)
                        return values, None
                    return [], None


                def _default_unflatten(values, context):
                    return values[0] if values else None


                pytree_module.__dict__['_model_output_flatten'] = _default_flatten
                pytree_module.__dict__['_model_output_unflatten'] = _default_unflatten
                saver.log_info("✓ Patched torch.utils._pytree.register_pytree_node for compatibility")
            else:
                saver.log_info("✓ torch.utils._pytree.register_pytree_node already exists")
        except Exception as e:
            saver.warning(f"Could not patch torch.utils._pytree: {e}")
            import traceback

            saver.warning(traceback.format_exc())
            # 继续执行，可能不需要这个补丁，或者会在导入时失败

        # ============ PyTorch Module.load_state_dict 兼容性补丁（支持 assign 参数）============
        # transformers 4.57.6 会调用 module.load_state_dict(..., assign=True)
        # 但 PyTorch 2.0.1 还不支持 assign 参数（是 PyTorch 2.1+ 的功能）
        # 所以我们需要修补 Module.load_state_dict 方法，让它接受并忽略 assign 参数
        try:
            import torch.nn as nn
            import inspect

            # 获取原始方法的绑定版本（unbound method）
            _orig_load_state_dict_unbound = nn.Module.load_state_dict.__func__ if hasattr(nn.Module.load_state_dict,
                                                                                          '__func__') else nn.Module.load_state_dict


            def _patched_load_state_dict(self, state_dict, strict=True, assign=None):
                """
                兼容版本的 load_state_dict，支持 assign 参数（但会忽略它）
                对于 assign=True 的情况，我们手动设置参数值
                """
                # 移除 assign 参数（PyTorch 2.0.1 不支持）
                # 如果 assign=True，我们需要手动设置参数值而不是使用 load_state_dict
                if assign:
                    # assign=True 意味着直接赋值，而不是加载
                    # 对于 PyTorch 2.0.1，我们手动设置每个参数
                    for name, param in self.named_parameters():
                        if name in state_dict:
                            param.data = state_dict[name].to(param.device).to(param.dtype)
                    for name, buffer in self.named_buffers():
                        if name in state_dict:
                            buffer.data = state_dict[name].to(buffer.device).to(buffer.dtype)
                    return None  # load_state_dict 在 strict=False 时返回 None
                else:
                    # assign=False 或 None，使用正常的 load_state_dict
                    # 尝试使用原始方法，但只传递 strict 参数
                    try:
                        # 使用 inspect 来安全地调用原始方法
                        sig = inspect.signature(_orig_load_state_dict_unbound)
                        if 'strict' in sig.parameters:
                            return _orig_load_state_dict_unbound(self, state_dict, strict=strict)
                        else:
                            # 如果原始方法不支持 strict，就不传
                            return _orig_load_state_dict_unbound(self, state_dict)
                    except Exception as e:
                        # 如果调用失败，手动实现基本功能
                        missing_keys = []
                        unexpected_keys = []
                        for name, param in self.named_parameters():
                            if name in state_dict:
                                param.data = state_dict[name].to(param.device).to(param.dtype)
                            else:
                                missing_keys.append(name)
                        for name in state_dict:
                            if name not in [n for n, _ in self.named_parameters()] and name not in [n for n, _ in
                                                                                                    self.named_buffers()]:
                                unexpected_keys.append(name)
                        if strict and (missing_keys or unexpected_keys):
                            error_msg = f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}"
                            raise RuntimeError(error_msg)
                        return None


            nn.Module.load_state_dict = _patched_load_state_dict
            saver.log_info("✓ Patched torch.nn.Module.load_state_dict to accept 'assign' parameter")
        except Exception as e:
            saver.warning(f"Could not patch Module.load_state_dict: {e}")
            import traceback

            saver.warning(traceback.format_exc())

        # 延迟导入 transformers，在 PyTorch 验证之后
        # 注意：不要在这里导入 AutoModelForMaskedLM，因为此时检测可能还没修复
        try:
            import transformers

            saver.log_info("Successfully imported transformers")
            saver.log_info(f"Transformers version: {transformers.__version__}")
        except ImportError as e:
            saver.error(f"Failed to import transformers: {e}")
            raise

        # 强制修复 transformers 的 PyTorch 检测
        # transformers 4.x 使用 import_utils 模块来检查后端
        try:
            import transformers.utils.import_utils as import_utils

            # 方法1: 直接设置 _torch_available
            if hasattr(import_utils, '_torch_available'):
                import_utils._torch_available = True
                saver.log_info("Set transformers.utils.import_utils._torch_available = True")

            # 方法2: 修改 _backends 字典（transformers 4.x 使用这个）
            # 注意：不要直接覆盖，而是确保 torch 在字典中
            if hasattr(import_utils, '_backends'):
                if 'torch' not in import_utils._backends or import_utils._backends['torch'] is None:
                    import_utils._backends['torch'] = torch
                    saver.log_info("Added/Updated torch in transformers._backends")
                else:
                    saver.log_info("torch already in transformers._backends")

            # 方法3: 直接修改 sys.modules，确保 transformers 能找到 torch
            if 'torch' not in sys.modules or sys.modules['torch'] is None:
                sys.modules['torch'] = torch
                saver.log_info("Updated sys.modules['torch']")

            # 验证修复是否成功
            if hasattr(transformers.utils, 'is_torch_available'):
                torch_available = transformers.utils.is_torch_available()
                saver.log_info(f"Transformers detects PyTorch (after fix): {torch_available}")
                if not torch_available:
                    saver.warning("Transformers still does not detect PyTorch after fix attempts")
            else:
                saver.warning("Transformers utils does not have is_torch_available method")

            # 修补 transformers.utils.generic 模块，确保 _model_output_flatten 等变量被定义
            # 这是因为我们的 register_pytree_node 补丁可能没有正确设置这些变量
            # 需要在模块导入时（而不是类定义时）定义这些变量
            try:
                # 先尝试导入 generic 模块（可能会失败，因为需要 register_pytree_node）
                # 如果导入失败，我们需要在导入前修补
                pass  # 将在下面处理
            except:
                pass

            # 在 transformers 导入后，立即修补 generic 模块
            # 这需要在任何使用 ModelOutput 的代码之前执行
            try:
                # 延迟导入 generic，因为它在导入时可能需要 register_pytree_node
                import transformers.utils.generic as generic_module

                # 检查是否需要修补
                # 注意：这些变量可能在模块的 __init_subclass__ 中定义，我们需要在类定义前提供
                if not hasattr(generic_module, '_model_output_flatten'):
                    # 提供兼容的函数实现
                    # 这些函数应该与 register_pytree_node 注册的函数匹配
                    def _model_output_flatten(output):
                        """兼容性补丁：为 ModelOutput 提供 flatten 函数"""
                        # 简单的实现：将 ModelOutput 转换为字典，然后展平
                        if hasattr(output, '__dict__'):
                            # 返回 (values, context) 元组
                            values = []
                            for key, value in output.__dict__.items():
                                if not key.startswith('_'):
                                    values.append(value)
                            return values, None
                        return [], None


                    def _model_output_unflatten(values, context):
                        """兼容性补丁：为 ModelOutput 提供 unflatten 函数"""
                        # 简单的实现：返回原始值（实际上不会被调用）
                        return values[0] if values else None


                    # 将这些函数添加到模块的命名空间中
                    generic_module.__dict__['_model_output_flatten'] = _model_output_flatten
                    generic_module.__dict__['_model_output_unflatten'] = _model_output_unflatten
                    saver.log_info("✓ Patched transformers.utils.generic with _model_output_flatten/unflatten")
                else:
                    saver.log_info("✓ transformers.utils.generic already has _model_output_flatten")
            except Exception as e:
                saver.warning(f"Could not patch transformers.utils.generic: {e}")
                import traceback

                saver.warning(traceback.format_exc())
                # 继续执行，可能不需要这个补丁，或者会在使用时失败

            # 注意：transformers 4.35.2 应该已经正确处理了设备映射
            # 我们不需要强制设置 map_location，因为这可能导致参数不兼容
            # 如果需要，可以通过 from_pretrained 的参数来控制设备
            saver.log_info(
                "✓ Skipping transformers.modeling_utils.load_state_dict patch (not needed for transformers 4.35.2)")

            # 在修复后重新导入 AutoModelForMaskedLM 和 AutoTokenizer
            # 注意：不重新加载模块（避免 PyTorch 版本兼容性问题），而是直接导入
            # 因为我们已经修复了检测机制，transformers 应该能正确识别 PyTorch
            try:
                # 直接导入（不重新加载模块，避免兼容性问题）
                from transformers import AutoModelForMaskedLM, AutoTokenizer

                saver.log_info(f"Imported AutoModelForMaskedLM: {type(AutoModelForMaskedLM)}")

                # 验证不是 DummyObject
                if 'DummyObject' in str(type(AutoModelForMaskedLM)):
                    saver.warning("AutoModelForMaskedLM is still a DummyObject, attempting manual fix...")

                    # 如果还是 DummyObject，尝试手动修复
                    # 方法1: 直接访问 transformers 的内部模块
                    try:
                        import transformers.models.auto.modeling_auto as modeling_auto_mod

                        # 检查模块是否有真实的类
                        if hasattr(modeling_auto_mod, 'AutoModelForMaskedLM'):
                            real_class = modeling_auto_mod.AutoModelForMaskedLM
                            if 'DummyObject' not in str(type(real_class)):
                                AutoModelForMaskedLM = real_class
                                saver.log_info(
                                    f"Got real AutoModelForMaskedLM from modeling_auto: {type(AutoModelForMaskedLM)}")
                            else:
                                raise ImportError("Real class is still DummyObject")
                        else:
                            raise ImportError("modeling_auto module does not have AutoModelForMaskedLM")
                    except Exception as e2:
                        saver.warning(f"Method 1 failed: {e2}, trying method 2...")
                        # 方法2: 直接使用 CodeBERT 的具体模型类（绕过 AutoModel）
                        try:
                            from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM

                            # CodeBERT 基于 RoBERTa，所以可以使用 RobertaForMaskedLM
                            AutoModelForMaskedLM = RobertaForMaskedLM
                            saver.log_info(f"Using RobertaForMaskedLM as fallback: {type(AutoModelForMaskedLM)}")
                        except Exception as e3:
                            saver.error(f"All manual fix methods failed. Method 1: {e2}, Method 2: {e3}")
                            raise ImportError(f"AutoModelForMaskedLM is DummyObject and all fix methods failed")
                else:
                    saver.log_info("AutoModelForMaskedLM is not a DummyObject, import successful")

            except Exception as e:
                saver.error(f"Failed to import AutoModelForMaskedLM: {e}")
                import traceback

                saver.error(traceback.format_exc())
                raise

        except Exception as e:
            saver.warning(f"Could not fix transformers PyTorch detection: {e}")
            import traceback

            saver.warning(traceback.format_exc())
            # 即使修复失败，也尝试导入（可能已经可以工作了）
            try:
                from transformers import AutoModelForMaskedLM, AutoTokenizer

                saver.log_info("Imported AutoModelForMaskedLM and AutoTokenizer (fallback)")
            except ImportError as e2:
                saver.error(f"Failed to import AutoModelForMaskedLM: {e2}")
                raise

        # 使用服务器路径（项目根目录为 /home/yutao/桌面/MPM/）
        # 如果设置了环境变量 CODEBERT_DIR，则使用环境变量；否则使用项目根目录下的 codebert 目录
        default_codebert_dir = join(get_root_path(), 'codebert')
        CODEBERT_DIR = os.getenv("CODEBERT_DIR", default_codebert_dir)
        CODEBERT_MODEL_ID = os.getenv("CODEBERT_MODEL_ID", "microsoft/codebert-base")
        saver.log_info(f"CodeBERT directory: {CODEBERT_DIR}")
        saver.log_info(f"CodeBERT model ID: {CODEBERT_MODEL_ID}")


        def _find_hf_model_path(base_dir: str) -> str:
            """查找 HuggingFace 下载的模型路径（可能在 snapshots 子目录下）"""
            base_path = Path(base_dir)

            # 检查根目录是否包含完整的模型文件（需要 config.json 和模型权重）
            if base_path.exists():
                has_model = (base_path / "pytorch_model.bin").exists() or (base_path / "model.safetensors").exists()
                has_config = (base_path / "config.json").exists()
                if has_model and has_config:
                    saver.log_info(f"Found complete model in root directory: {base_path}")
                    return str(base_path)

            # 递归查找 models--* 目录下的 snapshots，收集所有有效的路径
            valid_paths = []
            for model_dir in base_path.glob("models--*/snapshots/*"):
                if model_dir.is_dir():
                    has_model = (model_dir / "pytorch_model.bin").exists() or (model_dir / "model.safetensors").exists()
                    has_config = (model_dir / "config.json").exists()
                    if has_model and has_config:
                        valid_paths.append(model_dir)
                        saver.log_info(f"Found valid model path: {model_dir}")
                    elif has_model:
                        saver.warning(f"Found model weights in {model_dir} but missing config.json, skipping")

            if valid_paths:
                # 如果有多个有效路径，选择最新的（按修改时间）
                if len(valid_paths) > 1:
                    saver.warning(f"Found {len(valid_paths)} valid model paths, selecting the most recent one")
                    # 按 config.json 的修改时间排序，选择最新的
                    valid_paths.sort(key=lambda p: (p / "config.json").stat().st_mtime, reverse=True)
                selected_path = valid_paths[0]
                saver.log_info(f"Selected model path: {selected_path}")
                return str(selected_path)

            saver.warning(f"No valid model path found in {base_dir} (need both config.json and model weights)")
            return None


        # 验证 AutoModelForMaskedLM 是否正确导入
        if AutoModelForMaskedLM is None:
            saver.error("AutoModelForMaskedLM is None after import!")
            raise ImportError("AutoModelForMaskedLM is None")
        saver.log_info(f"AutoModelForMaskedLM type: {type(AutoModelForMaskedLM)}")

        model_path = _find_hf_model_path(CODEBERT_DIR)
        model_path = CODEBERT_DIR

        # 检查 AutoModelForMaskedLM 是否可用（不是 DummyObject）
        use_fallback = 'DummyObject' in str(type(AutoModelForMaskedLM))
        if use_fallback:
            saver.warning("AutoModelForMaskedLM is DummyObject, using RobertaForMaskedLM as fallback")
            try:
                from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM

                ModelClass = RobertaForMaskedLM
            except ImportError as e:
                saver.error(f"Failed to import RobertaForMaskedLM: {e}")
                raise
        else:
            ModelClass = AutoModelForMaskedLM

        # 验证 ModelClass 是否正确
        if ModelClass is None:
            saver.error("ModelClass is None!")
            raise ValueError("ModelClass is None")
        if not hasattr(ModelClass, 'from_pretrained'):
            saver.error(f"ModelClass {ModelClass} does not have from_pretrained method!")
            raise AttributeError(f"ModelClass {ModelClass} does not have from_pretrained method")
        saver.log_info(f"Using ModelClass: {ModelClass}, has from_pretrained: {hasattr(ModelClass, 'from_pretrained')}")

        # 尝试加载模型：先尝试本地路径，失败则使用模型 ID
        model_loaded = False
        if model_path:
            try:
                saver.log_info(f"Loading CodeBERT from local path: {model_path}")
                model = ModelClass.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=False
                ).to(FLAGS.device)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                saver.log_info(f"Loaded CodeBERT from local dir: {model_path}")
                model_loaded = True
            except Exception as e:
                saver.warning(f"Failed to load model from local path {model_path}: {e}")
                saver.warning("Falling back to using model ID (will download if needed)...")
                model_path = None  # 标记为失败，使用模型 ID

        # 如果本地路径加载失败或不存在，使用模型 ID
        if not model_loaded:
            # 使用模型ID下载（会下载到 cache_dir=CODEBERT_DIR）
            if model_path is None:
                saver.warning(
                    f"CodeBERT weights not found under {CODEBERT_DIR} or loading failed. "
                    f"Falling back to model id '{CODEBERT_MODEL_ID}' (will download to cache_dir)."
                )
            try:
                saver.log_info(f"Loading CodeBERT from model ID: {CODEBERT_MODEL_ID}")
                model = ModelClass.from_pretrained(
                    CODEBERT_MODEL_ID,
                    cache_dir=CODEBERT_DIR,
                    low_cpu_mem_usage=False
                ).to(FLAGS.device)
                tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL_ID, cache_dir=CODEBERT_DIR)
                saver.log_info(f"Downloaded and loaded CodeBERT model: {CODEBERT_MODEL_ID}")
            except Exception as e:
                saver.error(f"Failed to load model from HuggingFace: {e}")
                saver.error("This might be a transformers/PyTorch compatibility issue")
                import traceback

                saver.error(traceback.format_exc())
                raise

        # ==================== 为每个kernel生成最终数据 ====================
        # 遍历所有已处理的图，生成PyTorch Geometric的Data对象并保存
        if not all_gs:
            saver.error("No graphs found in all_gs! Cannot generate dataset.")
            raise ValueError("all_gs is empty, cannot proceed with dataset generation")

        saver.log_info(f"Generating dataset for {len(all_gs)} kernels")
        for gname, g in all_gs.items():
            # 创建边索引（PyTorch Geometric格式）
            edge_index = create_edge_index(g)
            saver.log_info('edge_index created', gname)
            data_list = []  # 存储当前kernel的所有Data对象

            # ==================== 确定保存路径 ====================
            # 根据kernel所属的benchmark设置不同的保存路径
            if gname in MACHSUITE_KERNEL:
                SAVE_DIR = join(graph_path, 'machsuite', gname)  # 图数据保存路径
                SAVE_DIR_1 = join(code_path, 'machsuite', gname)  # 代码特征保存路径
                CODE_FILES = join(get_root_path(), 'dse_database', 'programl', 'machsuite', gname, gname + '.c')
            else:
                SAVE_DIR = join(graph_path, 'poly', gname)
                SAVE_DIR_1 = join(code_path, 'poly', gname)
                CODE_FILES = join(get_root_path(), 'dse_database', 'programl', 'poly', gname, gname + '.c')

            # ==================== 读取源代码文件 ====================
            code_list = []
            try:
                with open(CODE_FILES, 'r', encoding='utf-8') as file:
                    fc = file.read()  # 读取原始源代码
            except FileNotFoundError:
                saver.error(f"Source code file not found: {CODE_FILES}")
                raise
            except Exception as e:
                saver.error(f"Failed to read source code file {CODE_FILES}: {e}")
                raise

            # ==================== 创建代码特征保存目录 ====================
            create_dir_if_not_exists(SAVE_DIR_1)

            # ==================== 处理每个配置变体 ====================
            # 检查是否有变体数据
            if not g.variants:
                saver.warning(f"No variants found for kernel {gname}, skipping...")
                continue

            # 收集所有设计点，用于保存 points_list.pkl
            points_list = []
            points_collected = 0
            points_none_count = 0
            for nums, (vname, d) in enumerate(g.variants.items()):
                d_node, d_edge, point = d  # 解包节点特征、边特征和设计点
                # 收集设计点
                points_list.append(point)
                points_collected += 1
                if point is None:
                    points_none_count += 1
                elif nums < 3:  # 只打印前3个样本的调试信息
                    saver.log_info(
                        f'[Debug] Collected point for {gname} variant {vname}: {type(point)}, keys: {list(point.keys()) if isinstance(point, dict) else "N/A"}')

            saver.log_info(
                f'[Debug] Collected {points_collected} points for kernel {gname} (None: {points_none_count}, non-None: {points_collected - points_none_count})')

            # 重新遍历 variants 来构建 Data 对象（因为 points_list 已经在上面收集完成）
            for nums, (vname, d) in enumerate(g.variants.items()):
                d_node, d_edge, point = d  # 解包节点特征、边特征和设计点

                # ==================== 转换为PyTorch张量 ====================
                # 使用编码器将特征字典转换为张量格式
                X = _encode_X_torch(d_node, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)
                edge_attr = _encode_edge_torch(d_edge, enc_ftype_edge, enc_ptype_edge)

                # ==================== 构建PyTorch Geometric Data对象 ====================
                # 包含图结构、节点特征、边特征和目标值
                if FLAGS.task == 'regression':
                    data_list.append(Data(
                        x=X,  # 节点特征矩阵
                        edge_index=edge_index,  # 边索引（COO格式）
                        perf=d_node['perf'],  # 归一化的性能目标
                        actual_perf=d_node['actual_perf'],  # 实际性能值
                        quality=d_node['quality'],  # 质量目标
                        util_BRAM=d_node['util-BRAM'],  # BRAM利用率
                        util_DSP=d_node['util-DSP'],  # DSP利用率
                        util_LUT=d_node['util-LUT'],  # LUT利用率
                        util_FF=d_node['util-FF'],  # FF利用率
                        total_BRAM=d_node['total-BRAM'],  # BRAM总量
                        total_DSP=d_node['total-DSP'],  # DSP总量
                        total_LUT=d_node['total-LUT'],  # LUT总量
                        total_FF=d_node['total-FF'],  # FF总量
                        edge_attr=edge_attr,  # 边特征
                        kernel=gname  # kernel名称
                    ))
                else:
                    raise NotImplementedError()

                # ==================== 生成代码特征 ====================
                # 将配置点的pragma值替换到源代码模板中
                fcc = deepcopy(fc)  # 复制源代码
                # 解析配置键，提取pragma值
                kd = [i for i in vname[4:].split('.')]  # 跳过前4个字符，按'.'分割
                kd_d = {}
                for j in kd:
                    jl = j.split('-')  # 按'-'分割，格式为"pragma_name-value"
                    if jl[1] == 'NA':
                        kd_d[jl[0]] = ''  # NA表示无值
                    else:
                        kd_d[jl[0]] = jl[1]  # 设置pragma值
                # 替换源代码中的pragma占位符
                for k, v in kd_d.items():
                    fcc = fcc.replace('auto' + '{' + k + '}', v)

                # ==================== CodeBERT编码 ====================
                # 使用CodeBERT对替换后的源代码进行编码
                try:
                    code_inputs = tokenizer(
                        fcc,
                        padding=True,
                        truncation=True,
                        max_length=512,  # 最大长度512
                        return_tensors="pt"
                    ).to(FLAGS.device)

                    # 确保模型在正确的设备上
                    model_device = next(model.parameters()).device
                    # 比较设备字符串表示，避免类型不匹配的警告
                    # 如果设备不同，才打印警告并移动输入
                    if str(model_device) != str(FLAGS.device):
                        saver.warning(
                            f"Model device ({model_device}) != FLAGS.device ({FLAGS.device}), moving inputs to model device")
                    # 始终将输入移动到模型所在的设备（静默处理，不打印警告）
                    code_inputs = {k: v.to(model_device) for k, v in code_inputs.items()}

                    # 获取所有层的隐藏状态
                    with torch.no_grad():  # 推理时不需要梯度
                        code_outputs = model(**code_inputs, output_hidden_states=True)

                    features = []
                    # 提取每层的[CLS] token特征（第一个token）
                    if hasattr(code_outputs, 'hidden_states') and code_outputs.hidden_states is not None:
                        for i, hidden_states in enumerate(code_outputs.hidden_states):
                            features.append(hidden_states[:, 0, :])
                        # 对所有层的特征求平均
                        code_features = torch.stack(features).sum(dim=0) / len(features)
                    else:
                        # 如果没有 hidden_states，使用最后一层的输出
                        saver.warning(
                            f"No hidden_states in CodeBERT output for {gname} variant {vname}, using last_hidden_state")
                        code_features = code_outputs.last_hidden_state[:, 0, :]

                    # 保存代码特征
                    pa = join(SAVE_DIR_1, f'{nums + 1}.pt')
                    torch.save(code_features.cpu(), pa)  # 保存到 CPU 以节省显存
                except Exception as e:
                    saver.error(f"Failed to encode code for {gname} variant {vname}: {e}")
                    import traceback

                    saver.error(traceback.format_exc())
                    # 不抛出异常，继续处理下一个样本，但记录错误
                    # 注意：代码特征对双塔模型很重要，但图数据仍然可以保存
                    saver.warning(f"Skipping code feature for {gname} variant {vname}, but continuing with graph data")

            # ==================== 保存数据到磁盘 ====================
            # 使用 try-finally 确保即使出错也保存 points_list.pkl
            try:
                saver.log_info(f'Saving {len(data_list)} to disk {SAVE_DIR}; Deleting existing files')
                # 安全删除目录（如果存在）
                if osp.exists(SAVE_DIR):
                    rmtree(SAVE_DIR)
                create_dir_if_not_exists(SAVE_DIR)  # 创建目录
                # 逐个保存Data对象
                for i in tqdm(range(len(data_list))):
                    torch.save(data_list[i], osp.join(SAVE_DIR, 'data_{}.pt'.format(i)))
            finally:
                # ==================== 保存 points_list.pkl ====================
                # 保存设计点列表，用于因果模型训练
                # 使用 finally 确保即使保存 data_*.pt 失败，也保存 points_list.pkl
                # 保存到专门的 points 目录：two_tower_dataset/points/{benchmark}/{kernel}/points_list.pkl
                try:
                    # 确定 benchmark
                    if gname in MACHSUITE_KERNEL:
                        benchmark = 'machsuite'
                    else:
                        benchmark = 'poly'

                    # 构建保存路径
                    kernel_points_dir = join(points_path, benchmark, gname)
                    points_list_path = join(kernel_points_dir, 'points_list.pkl')

                    # 确保目录存在
                    create_dir_if_not_exists(kernel_points_dir)

                    # 验证 points_list 和 data_list 的长度是否一致
                    if len(points_list) != len(data_list):
                        saver.warning(
                            f'⚠️  points_list length ({len(points_list)}) != data_list length ({len(data_list)}) for kernel {gname}')
                        saver.warning(f'⚠️  This may cause index mismatch. Truncating points_list to match data_list.')
                        # 截断 points_list 以匹配 data_list 的长度
                        points_list = points_list[:len(data_list)]

                    # 验证 points_list 不为空
                    if len(points_list) == 0:
                        saver.warning(f'points_list is empty for kernel {gname}, skipping points_list.pkl save')
                    else:
                        # 保存 points_list
                        with open(points_list_path, 'wb') as f:
                            pickle.dump(points_list, f)

                        # 验证文件是否成功保存
                        if osp.exists(points_list_path):
                            file_size = osp.getsize(points_list_path)
                            non_none_count = sum(1 for p in points_list if p is not None)
                            saver.log_info(
                                f'✓ Saved {len(points_list)} design points to {points_list_path} (non-None: {non_none_count}, file size: {file_size} bytes)')
                            saver.log_info(f'✓ points_list.pkl location: {osp.abspath(points_list_path)}')
                        else:
                            saver.error(f'✗ Failed to save points_list.pkl: file does not exist after save operation')
                            saver.error(f'✗ Expected path: {osp.abspath(points_list_path)}')
                except Exception as e:
                    saver.error(f'✗ Failed to save points_list.pkl for kernel {gname}: {e}')
                    import traceback

                    saver.error(traceback.format_exc())
                    # 不抛出异常，继续处理其他 kernel

            # ==================== 统计信息输出 ====================
            # 统计节点数量
            nns = [d.x.shape[0] for d in data_list]
            print_stats(nns, 'number of nodes')
            # 统计平均度数
            ads = [d.edge_index.shape[1] / d.x.shape[0] for d in data_list]
            print_stats(ads, 'avg degrees')
            saver.log_info('dataset[0].num_features', data_list[0].num_features)

            # ==================== 目标值分布分析 ====================
            # 对每个目标变量绘制分布图并统计
            for target in TARGETS:
                if not hasattr(data_list[0], target.replace('-', '_')):
                    saver.warning(f'Data does not have attribute {target}')
                    continue
                # 提取所有样本的目标值
                ys = [getattr(d, target.replace('-', '_')).item() for d in data_list]
                # if target == 'quality':
                #     continue
                # 绘制分布图
                plot_dist(ys, f'{target}_ys', saver.get_log_dir(), saver=saver, analyze_dist=True, bins=None)
                saver.log_info(f'{target}_ys', Counter(ys))


# ==================== 数据集类（用于训练时加载数据） ====================
class MyOwnDataset():
    """
    自定义数据集类，用于在训练时加载已生成的数据文件
    这个类提供了访问two_tower_dataset目录下数据的方法
    """

    def __init__(self):
        pass

    @property
    def raw_file_names(self):
        """
        返回原始文件列表（未使用）
        """
        # return ['some_file_1', 'some_file_2', ...]
        return []

    @property
    def processed_file_names(self):
        """
        返回所有已处理的数据文件路径
        返回两个字典：
        - gp: 图数据文件路径字典，key为kernel名称，value为文件路径列表
        - cp: 代码特征文件路径字典，key为kernel名称，value为文件路径列表
        """
        gp = {}  # graph paths
        cp = {}  # code paths
        for k in ALL_KERNEL:
            if k in MACHSUITE_KERNEL:
                gt = join(graph_path, 'machsuite', k)
                ct = join(code_path, 'machsuite', k)
            else:
                gt = join(graph_path, 'poly', k)
                ct = join(code_path, 'poly', k)
            gp[k] = glob(join(gt, '*.pt'))  # 获取所有.pt图数据文件
            cp[k] = glob(join(ct, '*.pt'))  # 获取所有.pt代码特征文件
        return gp, cp

    def download(self):
        """
        下载数据（未实现，因为数据是本地生成的）
        """
        pass

    # Download to `self.raw_dir`.

    def process(self):
        """
        处理原始数据（未实现，因为数据已经在gen_dataset.py中处理完成）
        """
        # i = 0
        # for raw_path in self.raw_paths:
        #     # Read data from `raw_path`.
        #     data = Data(...)
        #
        #     if self.pre_filter is not None and not self.pre_filter(data):
        #         continue
        #
        #     if self.pre_transform is not None:
        #         data = self.pre_transform(data)
        #
        #     torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
        #     i += 1
        pass

    def len(self):
        """
        返回数据集长度（未实现）
        """
        pass

    def __len__(self):
        return self.len()

    def get(self, idx):
        """
        获取指定索引的数据（未实现）
        """
        pass

    @staticmethod
    def get_data(idx, k):
        """
        静态方法：根据索引和kernel名称加载数据

        参数:
            idx: 数据索引
            k: kernel名称

        返回:
            data: 图数据（PyTorch Geometric Data对象）
            code_d: 代码特征（张量）
        """
        # 根据kernel所属的benchmark确定路径
        if k in MACHSUITE_KERNEL:
            gt = join(graph_path, 'machsuite', k)
            ct = join(code_path, 'machsuite', k)
        else:
            gt = join(graph_path, 'poly', k)
            ct = join(code_path, 'poly', k)
        # 加载图数据和代码特征
        data = torch.load(osp.join(gt, 'data_{}.pt'.format(idx)))
        code_d = torch.load(join(ct, f'{idx}.pt'))
        return data, code_d
