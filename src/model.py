
import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Module, Dropout, LayerNorm, Identity
import torch.nn.functional as F
from typing import Tuple
import numpy as np
import torch.nn as nn
from CoGNN.model_parse import GumbelArgs, EnvArgs, ActionNetArgs
from CoGNN.action_gumbel_layer import TempSoftPlus, ActionNet
from config import FLAGS
from src.utils import MLP, _get_y_with_target
from collections import OrderedDict, defaultdict
from nn_att import MyGlobalAttention
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool
from CoGNN.model_parse import GumbelArgs, EnvArgs, ActionNetArgs, ActivationType
from typing import NamedTuple, Any, Callable
from comp_model import GNN_DSE, HGP, Ironman, pna, progsg
from torch_geometric.utils import degree

def gin_mlp_func() -> Callable:
    def mlp_func(in_channels: int, out_channels: int, bias: bool):
        return Sequential(Linear(in_channels, out_channels, bias=bias),
                ReLU(), Linear(out_channels, out_channels, bias=bias))
    return mlp_func

gin_mlp_func = gin_mlp_func()

# comparative 子网直接吃 ``data.x``，维度须与 ProgramL/CDFG 构图一致（与 CoGNN.first_MLP_node 的 153 相同）。
# config 里 ``--num_features`` 默认 778 面向其它实验，勿套在 ironman/HGP/pna/GNN_DSE 的首层 conv 上，否则会与 data.x 列数不匹配。
_COMPARATIVE_GRAPH_NODE_IN = 153


def _convert_model_type_if_needed(model_type_value):
    """argparse 得到 str；EnvArgs/GumbelArgs/ActionNetArgs 需要 CoGNN.layers.ModelType 枚举。"""
    if isinstance(model_type_value, str):
        from CoGNN.layers import ModelType

        return ModelType.from_string(model_type_value)
    return model_type_value


_temp_mt = _convert_model_type_if_needed(FLAGS.temp_model_type)
_env_mt = _convert_model_type_if_needed(FLAGS.env_model_type)
_act_mt = _convert_model_type_if_needed(FLAGS.act_model_type)

gumbel_args = GumbelArgs(
    learn_temp=FLAGS.learn_temp,
    temp_model_type=_temp_mt,
    tau0=FLAGS.tau0,
    temp=FLAGS.temp,
    gin_mlp_func=gin_mlp_func,
)
env_args = EnvArgs(
    model_type=_env_mt,
    num_layers=FLAGS.env_num_layers,
    env_dim=FLAGS.env_dim,
    layer_norm=FLAGS.layer_norm,
    skip=FLAGS.skip,
    batch_norm=FLAGS.batch_norm,
    dropout=FLAGS.dropout,
    in_dim=FLAGS.num_features,
    out_dim=FLAGS.D,
    dec_num_layers=FLAGS.dec_num_layers,
    gin_mlp_func=gin_mlp_func,
    act_type=ActivationType.RELU,
)
action_args = ActionNetArgs(
    model_type=_act_mt,
    num_layers=FLAGS.act_num_layers,
    hidden_dim=FLAGS.act_dim,
    dropout=FLAGS.dropout,
    act_type=ActivationType.RELU,
    gin_mlp_func=gin_mlp_func,
    env_dim=FLAGS.env_dim,
)

class CoGNN(Module):
    def __init__(self, gumbel_args: GumbelArgs, env_args: EnvArgs, action_args: ActionNetArgs):
        super(CoGNN, self).__init__()
        self.task = FLAGS.task
        self.target = FLAGS.target
        self.D = FLAGS.D
        self.env_args = env_args
        self.learn_temp = gumbel_args.learn_temp
        self.first_MLP_env_attr = MLP(7, env_args.env_dim, activation_type=FLAGS.activation)
        self.first_MLP_act_attr = MLP(7, action_args.hidden_dim, activation_type=FLAGS.activation)
        self.first_MLP_node = MLP(153, env_args.env_dim, activation_type=FLAGS.activation)
        if gumbel_args.learn_temp:
            self.temp_model = TempSoftPlus(gumbel_args=gumbel_args, env_dim=env_args.env_dim)
        self.temp = gumbel_args.temp

        self.num_layers = env_args.num_layers
        self.env_net = env_args.load_net()

        layer_norm_cls = LayerNorm if env_args.layer_norm else Identity
        self.hidden_layer_norm = layer_norm_cls(env_args.env_dim)
        self.skip = env_args.skip
        self.dropout = Dropout(p=env_args.dropout)
        self.drop_ratio = env_args.dropout
        self.act = env_args.act_type.get()
        self.in_act_net = ActionNet(action_args=action_args)
        self.out_act_net = ActionNet(action_args=action_args)

        self.gate_nn = nn.Sequential(nn.Linear(self.D, self.D), ReLU(), Linear(self.D, self.D))
        self.glob = MyGlobalAttention(self.gate_nn, None)



    def forward(self, data):
        x, edge_index, edge_attr, batch = \
            data.x, data.edge_index, data.edge_attr, data.batch
        if hasattr(data, 'kernel'):
            gname = data.kernel[0]
        env_edge_attr = self.first_MLP_env_attr(edge_attr)
        act_edge_attr = self.first_MLP_act_attr(edge_attr)
        x = self.first_MLP_node(x)
        for gnn_idx in range(self.num_layers):
            x = self.hidden_layer_norm(x)

            # action
            in_logits = self.in_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_attr,
                                        act_edge_attr=act_edge_attr)
            out_logits = self.out_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_attr,
                                          act_edge_attr=act_edge_attr)
            temp = self.temp_model(x=x, edge_index=edge_index,
                                   edge_attr=env_edge_attr) if self.learn_temp else self.temp
            in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=True)
            edge_weight = self.create_edge_weight(edge_index=edge_index,
                                                  keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])

            # environment
            out = self.env_net[0 + gnn_idx](x=x, edge_index=edge_index, edge_weight=edge_weight,
                                            edge_attr=env_edge_attr)
            out = self.dropout(out)
            out = self.act(out)
            if self.skip:
                x = x + out
            else:
                x = out
        x = self.hidden_layer_norm(x)
        x = self.env_net[-1](x)  # decoder
        graph_emb = x
        out, node_att_scores = self.glob(x, batch)

        return graph_emb, out



    def create_edge_weight(self, edge_index: Adj, keep_in_prob: Tensor, keep_out_prob: Tensor) -> Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob


class InteractiveFusionBlock(nn.Module):

    def __init__(self, code_dim, graph_dim, hidden_dim):
        super().__init__()
        # 跨模态注意力机制
        self.graph_attn = nn.MultiheadAttention(graph_dim, num_heads=4)

        self.mp1 = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 特征转换层
        self.code_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.graph_transform = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU()
        )

        # 自适应门控
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, code_feats, graph_feats):
        code_feats = torch.tensor([item.cpu().detach().numpy() for item in code_feats]).cuda()
        code_feats = code_feats.squeeze(1)
        # 特征空间变换
        code_feats = self.mp1(code_feats)
        attn_graph, _ = self.graph_attn(
            query=graph_feats,
            key=code_feats,
            value=code_feats
        )

        trans_code = self.code_transform(code_feats)
        trans_graph = self.graph_transform(attn_graph)

        # 自适应门控融合
        combined = torch.cat([trans_code, trans_graph], dim=-1)
        gate = self.gate_network(combined)
        fused = gate * trans_graph + (1 - gate) * trans_code

        return fused, trans_code, trans_graph

import os.path as osp

class Net(nn.Module):
    def __init__(self, deg=0, edge_dim=0,
                 gnn_dim=FLAGS.D,
                 num_blocks=5):
        super().__init__()
        self.D = FLAGS.D
        self.task = FLAGS.task
        self.target = FLAGS.target
        self.device = FLAGS.device
        # 源代码编码器
        code_dim = 768
        if FLAGS.comparative_if:
            if FLAGS.comparative_model == 'HGP':
                self.HGP = HGP.HierNet(
                    in_channels=_COMPARATIVE_GRAPH_NODE_IN,
                    hidden_channels=FLAGS.hidden_num,
                    num_layers=3,
                    conv_type='sage',
                    drop_out=0.0,
                )
            elif FLAGS.comparative_model == 'ironman':
                self.ironman = Ironman.GCNNet(in_channels=_COMPARATIVE_GRAPH_NODE_IN)
            elif FLAGS.comparative_model == 'pna':
                self.pna = pna.PNANet(
                    in_dim=_COMPARATIVE_GRAPH_NODE_IN,
                    deg=deg,
                    num_layer=2,
                    emb_dim=200,
                    edge_dim=edge_dim,
                    drop_ratio=0.5,
                )
            elif FLAGS.comparative_model in ('progsg', 'ProgSG', 'PROGSG'):
                self.progsg_encoder = progsg.ProgSGNet(
                    in_channels=_COMPARATIVE_GRAPH_NODE_IN,
                    edge_dim=edge_dim if edge_dim else getattr(FLAGS, 'progsg_edge_dim', 7),
                )
            else:
                self.gnn_dse = GNN_DSE.Net(in_channels=_COMPARATIVE_GRAPH_NODE_IN)
        else:
            self.CoGNN = CoGNN(gumbel_args=gumbel_args, env_args=env_args, action_args=action_args)

        # 交互式融合网络 (分层设计)
        self.fusion_blocks = nn.ModuleList()
        fusion_dim = FLAGS.D # 初始融合维度
        for i in range(num_blocks):
            block = InteractiveFusionBlock(
                code_dim=code_dim,
                graph_dim=gnn_dim,
                hidden_dim=fusion_dim
            )
            self.fusion_blocks.append(block)

        self.loss_fucntion = torch.nn.MSELoss()

        if FLAGS.ablation_stu == 'LM':
            self.MLP1 = nn.Sequential(
                nn.Linear(768, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim)
            )
        self.MLPs = nn.ModuleDict()
        if 'regression' in self.task:
            _target_list = self.target
            if not isinstance(FLAGS.target, list):
                _target_list = [self.target]
            self.target_list = [t for t in _target_list]
        else:
            self.target_list = ['perf']
        d = self.D
        if d > 64:
            hidden_channels = [d // 2, d // 4, d // 8, d // 16, d // 32]
        else:
            hidden_channels = [d // 2, d // 4, d // 8]
        for target in self.target_list:
            self.MLPs[target] = MLP(d, FLAGS.out_dim, activation_type=FLAGS.activation,
                                        hidden_channels=hidden_channels,
                                        num_hidden_lyr=len(hidden_channels))

    def forward(self, data, code_f):
        if FLAGS.comparative_if:
            if FLAGS.comparative_model == 'HGP':
                graph_nodes_emb = self.HGP(data)
            elif FLAGS.comparative_model == 'ironman':
                graph_nodes_emb = self.ironman(data)
            elif FLAGS.comparative_model == 'pna':
                graph_nodes_emb = self.pna(data)
            elif FLAGS.comparative_model in ('progsg', 'ProgSG', 'PROGSG'):
                graph_nodes_emb = self.progsg_encoder(data)
            else:
                graph_nodes_emb = self.gnn_dse(data)
        else:
            graph_emb, graph_global = self.CoGNN(data)
            graph_nodes_emb = graph_global

        # 3. 分层交互融合
        fused_features = []
        for i, block in enumerate(self.fusion_blocks):
            fusion_input = (
                code_f,  # 选择对应的代码层
                graph_nodes_emb  # 切换图表示粒度
            )

            fused, code_feats, graph_feats = block(*fusion_input)
            fused_features.append(fused)
        if FLAGS.ablation_stu == 'CoGNN+LM':
            out = torch.mean(torch.stack(fused_features), dim=0)
        elif FLAGS.ablation_stu == 'CoGNN':
            out = graph_nodes_emb
        else:
            code_f = torch.tensor([item.cpu().detach().numpy() for item in code_f]).cuda()
            code_f = code_f.squeeze(1)
            code_f = self.MLP1(code_f)
            out = code_f

        out_dict = OrderedDict()
        total_loss = 0
        out_embed = out
        loss_dict = {}

        for target_name in self.target_list:
            # for target_name in target_list:
            out = self.MLPs[target_name](out_embed)
            y = _get_y_with_target(data, target_name)
            # 计算回归任务loss结果
            if self.task == 'regression':
                target = y.view((len(y), FLAGS.out_dim))
                if FLAGS.loss == 'RMSE':
                    loss = torch.sqrt(self.loss_fucntion(out, target))
                elif FLAGS.loss == 'MSE':
                    loss = self.loss_fucntion(out, target)
                else:
                    raise NotImplementedError()
            # 计算分类任务loss结果
            else:
                target = y.view((len(y)))
                loss = self.loss_fucntion(out, target)
            out_dict[target_name] = out
            total_loss += loss
            loss_dict[target_name] = loss
        return out_dict, total_loss, loss_dict