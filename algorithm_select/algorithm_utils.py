import torch


def extract_ast_meta_features(ast_json):
    """
    从AST中自动扫描代码的“物理基因”，生成9维统计向量。
    该过程在训练/推理前自动执行，无需手动标注。
    """
    stats = {
        'for_loops': 0,  # 1. 计算密集度
        'if_branches': 0,  # 2. 逻辑复杂度
        'array_decls': 0,  # 3. 存储需求
        'time_pragmas': 0,  # 4. 时间优化约束 (pipeline/unroll)
        'memory_access_score': 0,  # 5. 存储访问压力（更直接影响 BRAM/Perf）
        'max_loop_depth': 0,  # 6. 嵌套深度 (时延关键)
        'op_complexity': 0,  # 7. 高级运算 (mul/div/sqrt)
        'bit_width_score': 0,  # 8. 位宽敏感度 (ap_int/float/double)
        'total_nodes': 0  # 9. 整体规模
    }

    def contains_any(text, keywords):
        return any(k in text for k in keywords)

    def traverse(node, current_loop_depth):
        if not node or not isinstance(node, dict):
            return
        stats['total_nodes'] += 1

        # 提取当前 AST 的主要可用字段（兼容不同版本）
        n_type = str(node.get('type', '')).lower()
        n_text = str(node.get('text', '')).lower()
        n_pragma = str(node.get('pragma_full_text', '')).lower()
        n_name = str(node.get('name', '')).lower()
        n_val = str(node.get('value', '')).lower()
        merged_text = " ".join([n_text, n_pragma, n_name, n_val])

        # 1 & 6. 循环与嵌套深度
        new_depth = current_loop_depth
        if contains_any(n_type, ['for', 'while', 'loop']):
            stats['for_loops'] += 1
            new_depth += 1
            stats['max_loop_depth'] = max(stats['max_loop_depth'], new_depth)

        # 2. 逻辑分支
        if contains_any(n_type, ['if', 'switch', 'case', 'condition']):
            stats['if_branches'] += 1

        # 3. 存储声明（面向当前 AST 语料的文本匹配）
        if contains_any(merged_text, ['array', '[', ']', 'buffer', 'ram', 'bram', 'ap_memory']):
            stats['array_decls'] += 1

        # 4 & 5. pragma 分类（优先使用 pragma_full_text）
        pragma_text = n_pragma if n_pragma else merged_text
        is_pragma_like = contains_any(pragma_text, ['#pragma', 'pragma', 'accel'])

        # 时间相关 pragma：仍保留 pragma 门槛，避免普通文本误触发
        if is_pragma_like and contains_any(pragma_text, ['pipeline', 'unroll', 'parallel', 'ii']):
            stats['time_pragmas'] += 1

        # 5. 存储访问压力：统计“会放大带宽/存储消耗”的线索
        # 设计目的：比资源 pragma 更稳健，对 BRAM 与 Perf 更敏感
        memory_keywords = [
            '[', ']', 'load', 'store', 'memcpy', 'burst',
            'buffer', 'cache', 'array', 'ram', 'bram', 'uram',
            'm_axi', 's_axilite', 'ap_memory'
        ]
        if contains_any(merged_text, memory_keywords):
            stats['memory_access_score'] += 1

        # 7. 运算复杂度（基于文本和类型双通道）
        if contains_any(n_type, ['mul', 'div', 'sqrt']) or contains_any(merged_text, ['*', '/', '%', 'sqrt', 'pow']):
            stats['op_complexity'] += 1

        # 8. 位宽敏感度
        # 优先识别 HLS 常见位宽类型，其次识别通用数值宽度
        if contains_any(merged_text, ['ap_int', 'ap_uint', 'int8_t', 'int16_t', 'int32_t', 'int64_t',
                                      'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t']):
            stats['bit_width_score'] += 2
        elif contains_any(merged_text, ['double', 'float', 'half']):
            stats['bit_width_score'] += 2
        elif contains_any(merged_text, ['64', '32', '16']):
            stats['bit_width_score'] += 1

        # 递归遍历
        children = node.get('children', [])
        if isinstance(children, list):
            for child in children:
                traverse(child, new_depth)

    traverse(ast_json, 0)

    # 构造 9 维 Tensor
    res = torch.tensor([
        stats['for_loops'], stats['if_branches'], stats['array_decls'],
        stats['time_pragmas'], stats['memory_access_score'], stats['max_loop_depth'],
        stats['op_complexity'], stats['bit_width_score'], stats['total_nodes']
    ], dtype=torch.float32)

    # log1p 缩放保证门控网络输入平稳

    return torch.log1p(res)
# -----------------------------------------------------------------------------------------------------------
