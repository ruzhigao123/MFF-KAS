"""
Causal-aware QoR Prediction Model
实现 soft causal weight learning 的核心模块
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import re


class PragmaEncoder(nn.Module):
    """
    将 DesignPoint 中的 pragma 编码为向量表示
    
    每个 pragma p_i = {type_i, scope_i, value_i} 被编码为 e_i
    """
    def __init__(self, 
                 pragma_type_dim: int = 32,
                 pragma_scope_dim: int = 32, 
                 pragma_value_dim: int = 32,
                 output_dim: int = 64,
                 max_pragmas: int = 20):
        super().__init__()
        self.max_pragmas = max_pragmas
        self.output_dim = output_dim
        
        # Pragma type embedding (pipeline, unroll, tile, etc.)
        # 常见类型列表
        self.known_types = ['PIPE', 'PARA', 'TILE', 'TILING', 'ARRAY_PARTITION', 'DEPENDENCE', 'RESOURCE']
        self.type_vocab_size = len(self.known_types) + 1  # +1 for unknown
        self.type_embedding = nn.Embedding(self.type_vocab_size, pragma_type_dim)
        
        # Pragma scope embedding (L0, L1, L2, function name, etc.)
        # 使用字符级或简单的字符串hash
        self.scope_embedding = nn.Embedding(100, pragma_scope_dim)  # 假设最多100种scope
        
        # Pragma value embedding
        # 对于数值型，使用线性层；对于字符串型，使用embedding
        self.value_num_proj = nn.Linear(1, pragma_value_dim)  # 数值型
        self.value_str_embedding = nn.Embedding(50, pragma_value_dim)  # 字符串型（如'flatten', 'off'）
        
        # 融合层：将 type, scope, value 拼接后投影
        self.fusion = nn.Sequential(
            nn.Linear(pragma_type_dim + pragma_scope_dim + pragma_value_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # 字符串到索引的映射（用于 value）
        self.value_str_map = {'flatten': 0, 'off': 1, 'on': 2, '': 3}
        self.value_str_map_size = len(self.value_str_map)
    
    def _parse_pragma_id(self, pragma_id: str) -> Tuple[str, str]:
        """
        解析 pragma ID，例如 '__PIPE__L1' -> ('PIPE', 'L1')
        """
        # 移除前后下划线
        cleaned = pragma_id.strip('_')
        # 分割类型和scope
        parts = cleaned.split('__')
        if len(parts) >= 2:
            ptype = parts[0]
            scope = '__'.join(parts[1:])
        else:
            ptype = parts[0] if parts else 'UNKNOWN'
            scope = 'GLOBAL'
        return ptype.upper(), scope
    
    def _get_type_idx(self, ptype: str) -> int:
        """获取 type 的索引"""
        if ptype in self.known_types:
            return self.known_types.index(ptype)
        return len(self.known_types)  # unknown
    
    def _get_scope_idx(self, scope: str) -> int:
        """将 scope 字符串映射到索引（简单hash）"""
        # 简单hash，取模
        return hash(scope) % 100
    
    def _encode_value(self, value) -> torch.Tensor:
        """编码 pragma value"""
        device = next(self.parameters()).device  # 获取模型所在的设备
        if isinstance(value, (int, float)):
            # 数值型：归一化后投影
            val_tensor = torch.tensor([[float(value) / 100.0]], device=device)  # 简单归一化
            return self.value_num_proj(val_tensor)
        elif isinstance(value, str):
            # 字符串型：embedding
            if value in self.value_str_map:
                idx = self.value_str_map[value]
            else:
                idx = self.value_str_map_size  # unknown
            return self.value_str_embedding(torch.tensor([idx], device=device))
        else:
            # 默认
            return self.value_num_proj(torch.tensor([[0.0]], device=device))
    
    def forward(self, design_point: Dict[str, any]) -> torch.Tensor:
        """
        将 DesignPoint 编码为 pragma embeddings
        
        Args:
            design_point: Dict[str, Union[int, str]]，例如 {'__PIPE__L1': 'flatten', '__TILE__L2': 4}
        
        Returns:
            pragma_embeddings: Tensor of shape [num_pragmas, output_dim]
        """
        device = next(self.parameters()).device  # 获取模型所在的设备
        pragma_embeddings: List[torch.Tensor] = []
        pragma_ids: List[Optional[str]] = []
        pragma_mask: List[int] = []

        # 为了可解释性/复现性，保证 pragma 顺序稳定
        items = sorted(design_point.items(), key=lambda x: x[0])
        # 截断到 max_pragmas
        items = items[: self.max_pragmas]

        for pragma_id, value in items:
            # 解析 type 和 scope
            ptype, scope = self._parse_pragma_id(pragma_id)

            # 获取 embeddings
            type_idx = self._get_type_idx(ptype)
            scope_idx = self._get_scope_idx(scope)

            type_emb = self.type_embedding(torch.tensor([type_idx], device=device))
            scope_emb = self.scope_embedding(torch.tensor([scope_idx], device=device))
            value_emb = self._encode_value(value)

            # 拼接并融合
            combined = torch.cat([type_emb, scope_emb, value_emb], dim=-1)
            pragma_emb = self.fusion(combined)  # [1, output_dim]
            pragma_embeddings.append(pragma_emb)
            pragma_ids.append(pragma_id)
            pragma_mask.append(1)

        # 补零到固定长度 max_pragmas，保证每个样本输出形状一致
        if len(pragma_embeddings) < self.max_pragmas:
            pad_n = self.max_pragmas - len(pragma_embeddings)
            zero_pad = torch.zeros(pad_n, self.output_dim, device=device)
            if len(pragma_embeddings) > 0:
                emb = torch.cat(pragma_embeddings, dim=0)  # [n, output_dim]
                emb = torch.cat([emb, zero_pad], dim=0)   # [max_pragmas, output_dim]
            else:
                emb = torch.cat([torch.zeros(0, self.output_dim, device=device), zero_pad], dim=0)
            pragma_ids.extend([None] * pad_n)
            pragma_mask.extend([0] * pad_n)
        else:
            # len == max_pragmas
            emb = torch.cat(pragma_embeddings, dim=0) if len(pragma_embeddings) > 0 else torch.zeros(self.max_pragmas, self.output_dim, device=device)
            if len(pragma_mask) < self.max_pragmas:
                pragma_mask.extend([0] * (self.max_pragmas - len(pragma_mask)))

        # 记录本次编码的 pragma 顺序（固定长度，pad 部分为 None）
        self._last_pragma_ids = pragma_ids
        # 记录 mask（pad 部分为 0），供 CausalHead 屏蔽 padding
        self._last_pragma_mask = torch.tensor(pragma_mask, device=device, dtype=torch.bool)
        return emb


class CausalHead(nn.Module):
    """
    实现 soft causal weight α_ij 和因果预测公式
    
    对每个 QoR 目标 y_j，学习 α_ij(C) 和 g_j(p_i)
    预测公式: y_j = f_j(C) + sum_i α_ij * g_j(p_i)
    """
    def __init__(self,
                 context_dim: int,
                 pragma_emb_dim: int,
                 num_targets: int,
                 hidden_dim: int = 64,
                 use_attention: bool = True):
        super().__init__()
        self.context_dim = context_dim
        self.pragma_emb_dim = pragma_emb_dim
        self.num_targets = num_targets
        self.use_attention = use_attention
        
        # Baseline 预测器 f_j(C)
        self.baseline_heads = nn.ModuleDict()
        for i in range(num_targets):
            self.baseline_heads[f'target_{i}'] = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        
        if use_attention:
            # 使用 attention 机制计算 α_ij
            # 对每个 target，有一个 query
            self.target_queries = nn.Parameter(torch.randn(num_targets, context_dim))
            # 将 pragma embedding 投影到与 query 相同的空间
            self.pragma_proj = nn.Linear(pragma_emb_dim, context_dim)
        else:
            # 使用 MLP 直接计算 α_ij
            self.alpha_mlps = nn.ModuleDict()
            for i in range(num_targets):
                self.alpha_mlps[f'target_{i}'] = nn.Sequential(
                    nn.Linear(context_dim + pragma_emb_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
        
        # Pragma 到 QoR 的投影 g_j(p_i)
        self.pragma_to_qor = nn.ModuleDict()
        for i in range(num_targets):
            self.pragma_to_qor[f'target_{i}'] = nn.Linear(pragma_emb_dim, 1)
    
    def forward(self, 
                context: torch.Tensor,
                pragma_embeddings: torch.Tensor,
                target_names: List[str],
                pragma_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算因果预测和 soft causal weights
        
        Args:
            context: [batch_size, context_dim] 程序上下文
            pragma_embeddings: [num_pragmas, pragma_emb_dim] pragma embeddings
            target_names: List[str] QoR 目标名称列表
        
        Returns:
            predictions: Dict[str, Tensor] 每个目标的预测值
            alpha_matrix: Tensor [batch_size, num_pragmas, num_targets] soft causal weights
        """
        batch_size = context.shape[0]
        num_pragmas = pragma_embeddings.shape[0]
        
        # 扩展 pragma_embeddings 到 batch 维度
        # [num_pragmas, pragma_emb_dim] -> [batch_size, num_pragmas, pragma_emb_dim]
        pragma_emb_batch = pragma_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        predictions = OrderedDict()
        alpha_list = []

        # pragma_mask: [num_pragmas] or [batch_size, num_pragmas] with True for valid pragmas
        if pragma_mask is not None:
            if pragma_mask.dim() == 1:
                pragma_mask_b = pragma_mask.unsqueeze(0).expand(batch_size, -1)
            else:
                pragma_mask_b = pragma_mask
            pragma_mask_b = pragma_mask_b.to(context.device)
        else:
            pragma_mask_b = None
        
        for target_idx, target_name in enumerate(target_names):
            # Baseline 预测 f_j(C)
            baseline = self.baseline_heads[f'target_{target_idx}'](context)  # [batch_size, 1]
            
            # 计算 α_ij
            if self.use_attention:
                # Query: [batch_size, context_dim]
                # 确保 target_queries 在正确的设备上
                target_query = self.target_queries[target_idx].unsqueeze(0).to(context.device)
                query = context + target_query  # [batch_size, context_dim]
                # Key: [batch_size, num_pragmas, context_dim]
                key = self.pragma_proj(pragma_emb_batch)
                # 计算 attention scores
                # α_ij = query^T * key
                alpha_raw = torch.bmm(
                    query.unsqueeze(1),  # [batch_size, 1, context_dim]
                    key.transpose(1, 2)   # [batch_size, context_dim, num_pragmas]
                ).squeeze(1)  # [batch_size, num_pragmas]
                # 对 padding 位置做 mask，避免 None/pad 影响解释与训练
                if pragma_mask_b is not None:
                    alpha_raw = alpha_raw.masked_fill(~pragma_mask_b, float('-inf'))
                # softmax 归一化：提升稳定性与可解释性（每个 target 权重和为 1）
                alpha = torch.softmax(alpha_raw, dim=-1)
            else:
                # 使用 MLP
                # 拼接 context 和 pragma_emb
                context_expanded = context.unsqueeze(1).expand(-1, num_pragmas, -1)  # [batch_size, num_pragmas, context_dim]
                combined = torch.cat([context_expanded, pragma_emb_batch], dim=-1)  # [batch_size, num_pragmas, context_dim + pragma_emb_dim]
                alpha_raw = self.alpha_mlps[f'target_{target_idx}'](combined).squeeze(-1)  # [batch_size, num_pragmas]
                if pragma_mask_b is not None:
                    alpha_raw = alpha_raw.masked_fill(~pragma_mask_b, float('-inf'))
                alpha = torch.softmax(alpha_raw, dim=-1)
            
            alpha_list.append(alpha)
            
            # 计算 g_j(p_i)
            pragma_contrib = self.pragma_to_qor[f'target_{target_idx}'](pragma_emb_batch)  # [batch_size, num_pragmas, 1]
            pragma_contrib = pragma_contrib.squeeze(-1)  # [batch_size, num_pragmas]
            
            # 加权求和: sum_i α_ij * g_j(p_i)
            weighted_contrib = (alpha.unsqueeze(-1) * pragma_contrib.unsqueeze(-1)).sum(dim=1)  # [batch_size, 1]
            
            # 最终预测
            prediction = baseline + weighted_contrib  # [batch_size, 1]
            # 保持 [batch_size, 1]，避免外部 loss 计算出现 broadcasting
            predictions[target_name] = prediction
        
        # 构建 alpha matrix: [batch_size, num_pragmas, num_targets]
        alpha_matrix = torch.stack(alpha_list, dim=-1)  # [batch_size, num_pragmas, num_targets]
        
        return predictions, alpha_matrix
