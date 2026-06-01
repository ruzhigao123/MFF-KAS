import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Linear, Dropout, ReLU, LayerNorm, Identity
from torch_geometric.nn import GlobalAttention
from torch_geometric.typing import Adj, Tensor
import os


class TripleModalFusionBlock(nn.Module):    # 三特征融合用的
    def __init__(self, g_dim=128, s_dim=768, a_dim=784, embed_dim=256, num_heads=8, dropout=0.1):
        super(TripleModalFusionBlock, self).__init__()

        # 1. 维度对齐层：将 CDFG(128), Text(768), AST(784) 统一映射到 embed_dim
        self.proj_g = nn.Linear(g_dim, embed_dim)
        self.proj_s = nn.Linear(s_dim, embed_dim)
        self.proj_a = nn.Linear(a_dim, embed_dim)

        # 2. 第一阶段交叉注意力：CDFG (Q) + Text (K, V)
        # 对应文档公式 (4)-(7)
        self.attn_gs = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        # 3. 第二阶段交叉注意力：Intermediate (Q) + AST (K, V)
        # 对应文档公式 (8)-(10)
        self.attn_gsa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 4. 门控融合层 (Gated Residual Connection)
        # 用于平衡原始图特征与融合后的特征
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, h_g, h_s, h_a):
        """
        h_g: [Batch, g_dim] -> 来自 GNN 的图特征
        h_s: [Batch, s_dim] -> 来自 CodeBERT 的文本特征
        h_a: [Batch, a_dim] -> 来自升级版 AST 脚本的特征 (784维)
        """
        # 统一映射维度并增加序列维度 [Batch, 1, embed_dim]
        g_emb = self.proj_g(h_g).unsqueeze(1)
        s_emb = self.proj_s(h_s).unsqueeze(1)
        a_emb = self.proj_a(h_a).unsqueeze(1)

        # --- 第一阶段：CDFG + Text ---
        # Q=CDFG, K=V=Text
        h_gs, _ = self.attn_gs(query=g_emb, key=s_emb, value=s_emb)
        h_gs = self.norm1(h_gs + g_emb)  # 残差连接

        # --- 第二阶段：(CDFG+Text) + AST ---
        # Q=h_gs, K=V=AST
        h_fuse, _ = self.attn_gsa(query=h_gs, key=a_emb, value=a_emb)
        h_fuse = self.norm2(h_fuse + h_gs)

        h_fuse = h_fuse.squeeze(1)  # [Batch, embed_dim]

        # --- 第三阶段：门控控制 (Gating) ---
        # 结合原始投影后的图特征与最终融合特征
        g_proj = g_emb.squeeze(1)
        gate_val = self.gate(torch.cat([g_proj, h_fuse], dim=-1))
        h_final = gate_val * g_proj + (1 - gate_val) * h_fuse

        return h_final



class GumbelArgs:
    def __init__(self, learn_temp=True, temp=1.0, tau0=1.0):
        self.learn_temp = learn_temp
        self.temp = temp
        self.tau0 = tau0


class EnvArgs:
    def __init__(self, num_layers=5, env_dim=128, dropout=0.2, in_dim=153):
        self.num_layers = num_layers
        self.env_dim = env_dim
        self.dropout = dropout
        self.in_dim = in_dim
        self.skip = True
        self.batch_norm = True
        self.layer_norm = True


class CoGNN(Module):
    def __init__(self, in_channels=153, env_dim=128, out_dim=256, num_layers=3, dropout=0.2):
        super(CoGNN, self).__init__()
        self.env_dim = env_dim
        self.num_layers = num_layers

        # 初始特征变换
        self.first_MLP_node = MLP(in_channels, env_dim)
        self.first_MLP_edge = MLP(7, env_dim)  # 假设edge_attr维度为7

        # 简化版环境变量网络 (env_net)
        # 在实际项目中，env_net 通常是多个 GNN 层（如 GIN, GAT 等）
        self.gnn_layers = nn.ModuleList([
            # 这里简单示例使用全连接或你可以替换为标准的 PyG 卷积层
            nn.Linear(env_dim, env_dim) for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(env_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.decoder = nn.Linear(env_dim, out_dim)

        # 注意力池化
        self.gate_nn = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
        self.glob = MyGlobalAttention(self.gate_nn, None)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 节点与边特征映射
        x = self.first_MLP_node(x)
        edge_feat = self.first_MLP_edge(edge_attr)

        # 多层演化 (简化 CoGNN 逻辑以适应推荐任务)
        for i in range(self.num_layers):
            identity = x
            out = self.gnn_layers[i](x)
            out = self.dropout(torch.relu(out))
            x = identity + out if i > 0 else out
            x = self.layer_norm(x)

        x = self.decoder(x)
        graph_emb, _ = self.glob(x, batch)
        return graph_emb


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, activation_type='relu'):
        super(MLP, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU() if activation_type == 'relu' else nn.Identity()

    def forward(self, x):
        return self.act(self.linear(x))


class MyGlobalAttention(nn.Module):
    def __init__(self, gate_nn, nn=None):
        super(MyGlobalAttention, self).__init__()
        self.glob = GlobalAttention(gate_nn, nn)

    def forward(self, x, batch):
        return self.glob(x, batch), None
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_p=0.3):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        # 替换为 LayerNorm，不再需要判断 batch size 是否为 1
        self.ln1 = nn.LayerNorm(dim)

        self.linear2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout_p)
        self.activation = nn.LeakyReLU(0.1)  # 使用 LeakyReLU 防止神经元死亡

    def forward(self, x):
        identity = x

        # 第一层卷积/全连接
        out = self.linear1(x)
        out = self.ln1(out)
        out = self.activation(out)

        # 第二层
        out = self.linear2(out)
        out = self.ln2(out)
        out = self.dropout(out)

        # 残差连接
        out += identity
        return self.activation(out)

# --- 2. 新增：Meta 与 AST 后 16 维特征拼接升维模块 ---

class CombinedMetaMLP(nn.Module):
    def __init__(self, meta_base_dim=9, ast_tail_dim=16, out_dim=128):
        super(CombinedMetaMLP, self).__init__()
        # 拼接后的初始维度：9 + 16 = 25 维
        combined_in_dim = meta_base_dim + ast_tail_dim

        self.net = nn.Sequential(
            nn.Linear(combined_in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, out_dim),  # 升维至 128 (或你需要的维度)
            nn.ReLU()
        )

    def forward(self, meta_base, ast_tail):
        # meta_base: [Batch, 9], ast_tail: [Batch, 16]
        x = torch.cat([meta_base, ast_tail], dim=1)
        return self.net(x)
from src.comp_model.GCN import GCN  # 文件是 pna.py, 类是 PNANet
class AlgorithmSelector(nn.Module):  # 三特征融合；logits 宽度 = num_tools
    """
    num_tools：最后一层每个指标 head 的输出维度（专家类数）。

    - 7：全量专家（labels 7 维顺序：0..6）。
    - 5：子集实验见 select_main 系列脚本传入的切片维数。
    - 3：三专家子集；仅全局 4,5,6 时传 num_tools=3（cdfg_only, ast_only, LM）。
    - 4：四专家扁平分类（旧版脚本）。
    - 两级结构（粗 only_ast/cdfg/text + fusion×细四类）：见 select_model_three_only_modalities.AlgorithmSelectorThreeOnlyModalities，
      不使用本类的 num_tools，而用两套各 4 维 logits。
    """

    def __init__(self, gcn_params=None, num_tools=7):
        super(AlgorithmSelector, self).__init__()
        self.num_tools = num_tools

        # 维度配置
        hidden_dim = gcn_params.get('hidden_channels', 128) if gcn_params else 128
        graph_out_dim = 128      # GCN 输出维度
        text_dim = 768           # CodeBERT 特征维度
        ast_high_dim = 784       # 升级版 AST 脚本特征维度
        fusion_embed_dim = 256   # 融合后的统一特征维度
        metric_dim = 32          # 任务 ID 嵌入维度



        # A. 图特征提取 (GCN)
        self.cognn = GCN(
            in_channels=153,
            hidden_channels=hidden_dim,
            num_layers=7,
            drop_out=0.28
        )

        # B. 【核心修改】三特征融合模块
        # 融合 CDFG (128), Text (768), AST (784) -> 256
        self.fusion_block = TripleModalFusionBlock(
            g_dim=graph_out_dim,
            s_dim=text_dim,
            a_dim=ast_high_dim,
            embed_dim=fusion_embed_dim,
            num_heads=8,
            dropout=0.1
        )

        meta_base_dim = 9
        ast_tail_dim = 16
        meta_combined_out_dim = 128
        self.meta_processor = CombinedMetaMLP(
            meta_base_dim=meta_base_dim,
            ast_tail_dim=ast_tail_dim,
            out_dim=meta_combined_out_dim
        )

        # C. 任务指示嵌入 (Metric Embedding)
        self.metric_emb = nn.Embedding(5, metric_dim)

        # D. Metric Gating (指标门控层)
        # 作用于融合后的 256 维特征
        self.gate_layer = nn.Sequential(
            nn.Linear(metric_dim, 128),
            nn.ReLU(),
            nn.Linear(128, fusion_embed_dim),
            nn.Sigmoid()
        )

        # E. 最终拼接后的总维度: 融合特征 (256) + 任务嵌入 (32) = 288
        total_input_dim = fusion_embed_dim + metric_dim

        self.target_names = ['perf', 'lut', 'ff', 'dsp', 'bram']
        self.heads = nn.ModuleDict({
            name: self._make_res_head(total_input_dim, num_tools)
            for name in self.target_names
        })

    def _make_res_head(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim)
        )

    def forward(self, data, h_s, h_a, meta_raw, m_id):
        """
        Args:
            data: PyG Batch 对象
            h_s: Text 特征 [Batch, 768]
            h_a: AST 特征 [Batch, 784]
            m_id: 指标 ID [Batch]
        """
        # 1. 提取图特征 [Batch, 128]
        graph_feat = self.cognn(data)

        # 2. 三特征融合 (CDFG + Text + AST) [Batch, 256]
        # 暂时跳过之前的 25 维元特征，直接使用高维特征融合
        fused_feat = self.fusion_block(graph_feat, h_s, h_a)

        # 3. 任务嵌入 [Batch, 32]
        task_feat = self.metric_emb(m_id)

        meta_base = meta_raw[:, :9]
        meta_tail = meta_raw[:, 9:]
        meta_out = self.meta_processor(meta_base, meta_tail)

        # 4. 执行 Metric Gating (对融合特征进行门控过滤)
        # gate_signals = self.gate_layer(task_feat)
        # gated_fused_feat = fused_feat * gate_signals

        # 5. 特征拼接 [Batch, 256 + 32 = 288]
        combined = torch.cat([fused_feat, task_feat], dim=1)

        # 6. 初始化输出张量（与 num_tools 一致；子集实验需与 labels 切片维数一致）
        final_logits = torch.zeros(combined.size(0), self.num_tools, device=combined.device, dtype=combined.dtype)

        # 7. 按照指标任务进行批量路由计算
        for target_idx, name in enumerate(self.target_names):
            mask = (m_id == target_idx)
            if mask.any():
                final_logits[mask] = self.heads[name](combined[mask])

        return final_logits  # , gate_signals


import torch
import torch.nn as nn
import torch.nn.functional as F