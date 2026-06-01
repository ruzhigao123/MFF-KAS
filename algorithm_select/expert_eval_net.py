# -*- coding: utf-8 -*-
"""
专家 RMSE 评估专用网络（与 ``src/model.py`` 的 ``Net`` 权重键兼容，前向支持七种专家）。

不修改 ``model.py`` / ``train.py``。由 ``expert_rmse_eval_train_split.apply_eval_mock_flags`` 写入
``config.FLAGS`` 的 ``expert_rmse_gnn_only`` / ``expert_eval_lm_only`` / ``expert_text_only_inference`` 等
控制分支；与训练用 ``Net`` 解耦，避免纯 GNN 样本误走融合支路缺少 ``text_feat``/``ast_feat``。
"""
from __future__ import annotations

import os
import sys

_alg_dir = os.path.dirname(os.path.abspath(__file__))
_base_dir = os.path.normpath(os.path.join(_alg_dir, ".."))
_src_dir = os.path.join(_base_dir, "src")
for _p in (_base_dir, _src_dir, _alg_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from config import FLAGS
from src.model import TripleModalFusionBlock
import src.model as _model_src
from src.utils import MLP, _get_y_with_target

from comp_model.GNN_DSE import Net as GNN_DSE_Net
from comp_model.HGP import HierNet
from comp_model.Ironman import GCNNet
from comp_model.pna import PNANet
from comp_model.GCN import GCN
from comp_model.ironman_no_edage import GCNNet_NO_Edage
from comp_model.GraphSAGE import GraphSAGE
from comp_model.GAT import GAT
from comp_model.GIN import GIN


class ExpertEvalNet(nn.Module):
    """
    与 ``src.model.Net`` 相同的 ``state_dict`` 键名（gnn_model / fusion_block / MLPs / MLP1），
    前向按 FLAGS 在「纯 GNN / 文本(LM) / 三模态融合」之间切换。
    """

    def __init__(self, in_channels, target_list, deg=None, task="regression"):
        super().__init__()
        self.in_channels = in_channels
        self.target_list = list(target_list)
        self.task = task

        self._expert_eval_lm_only = bool(getattr(FLAGS, "expert_eval_lm_only", False))
        self._expert_rmse_gnn_only = bool(getattr(FLAGS, "expert_rmse_gnn_only", False))
        self._train_lm = FLAGS.ablation_stu == "LM"
        self._expert_text_only_infer = bool(getattr(FLAGS, "expert_text_only_inference", False))

        _build_fusion = (
            not self._expert_rmse_gnn_only
            and not self._expert_eval_lm_only
            and not self._train_lm
        )
        if _build_fusion:
            self.fusion_block = TripleModalFusionBlock(
                g_dim=128,
                s_dim=768,
                a_dim=784,
                embed_dim=256,
            )
        else:
            self.fusion_block = None

        _need_mlp1 = self._train_lm or self._expert_eval_lm_only or self._expert_text_only_infer
        _mlp1_dim = FLAGS.hidden_num if self._expert_eval_lm_only else FLAGS.D
        if _need_mlp1:
            self.MLP1 = nn.Sequential(
                nn.Linear(768, _mlp1_dim),
                nn.ReLU(),
                nn.Linear(_mlp1_dim, _mlp1_dim),
            )
        else:
            self.MLP1 = None

        model_name = FLAGS.comparative_model.lower()
        if FLAGS.comparative_if:
            if model_name == "pna":
                self.gnn_model = PNANet(
                    in_dim=in_channels,
                    deg=deg,
                    num_layer=FLAGS.num_layers,
                    emb_dim=FLAGS.hidden_num,
                )
            elif model_name == "gnn-dse" or model_name == "gin":
                self.gnn_model = GNN_DSE_Net(in_channels, FLAGS.hidden_num)
            elif model_name == "hgp":
                self.gnn_model = HierNet(
                    in_channels, FLAGS.hidden_num, num_layers=3, conv_type="sage", drop_out=0.0
                )
            elif model_name == "ironman":
                self.gnn_model = GCNNet(in_channels=FLAGS.num_features)
            elif model_name == "gcn":
                self.gnn_model = GCN(
                    in_channels=in_channels,
                    hidden_channels=FLAGS.hidden_num,
                    num_layers=6,
                    drop_out=0.15,
                )
            elif model_name == "ironman_no_edge":
                self.gnn_model = GCNNet_NO_Edage(
                    in_channels=in_channels,
                    hidden_channels=FLAGS.D,
                    num_layers=6,
                    drop_out=0,
                )
            elif model_name == "GraphSAGE":
                self.gnn_model = GraphSAGE(
                    in_channels=in_channels,
                    hidden_channels=FLAGS.D,
                    num_layers=3,
                    drop_out=0.1,
                )
            elif model_name == "gin":
                self.gnn_model = GIN(
                    in_channels=in_channels,
                    hidden_channels=FLAGS.D,
                    num_layers=6,
                    drop_out=0.2,
                )
            elif model_name == "gat":
                self.gnn_model = GAT(
                    in_channels=in_channels,
                    hidden_channels=FLAGS.D,
                    num_layers=6,
                    drop_out=0.2,
                )
            else:
                self.gnn_model = GNN_DSE_Net(in_channels, FLAGS.hidden_num)
        else:
            self.gnn_model = _model_src.CoGNN(
                gumbel_args=_model_src.gumbel_args,
                env_args=_model_src.env_args,
                action_args=_model_src.action_args,
            )

        if self._expert_rmse_gnn_only or self._expert_eval_lm_only:
            _mlp_in = FLAGS.hidden_num
        elif self._train_lm:
            _mlp_in = FLAGS.D
        else:
            _mlp_in = 256

        self.MLPs = nn.ModuleDict()
        for target_name in self.target_list:
            self.MLPs[target_name] = MLP(
                input_dim=_mlp_in,
                output_dim=FLAGS.out_dim,
                activation_type="relu",
                num_hidden_lyr=2,
                hidden_channels=[FLAGS.hidden_num, FLAGS.hidden_num],
            )

        self.loss_fucntion = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()

    def forward(self, data, ablation_mode="all"):
        ex_gnn = bool(getattr(FLAGS, "expert_rmse_gnn_only", False))
        _lm_branch = (FLAGS.ablation_stu == "LM") or bool(
            getattr(FLAGS, "expert_text_only_inference", False)
        )

        if ex_gnn:
            out_embed = self.gnn_model(data)
            out_dict = {}
            loss_dict = {}
            total_loss = 0
            for target_name in self.target_list:
                out = self.MLPs[target_name](out_embed)
                out_dict[target_name] = out
                y = _get_y_with_target(data, target_name)
                if y is None:
                    continue
                target = y.view((len(y), FLAGS.out_dim))
                loss = self.loss_fucntion(out, target)
                total_loss += loss
                loss_dict[target_name] = loss
            return out_dict, total_loss, loss_dict

        if _lm_branch:
            tf = data.text_feat
            if tf.dim() == 1 and tf.numel() == 768:
                text_b = tf.unsqueeze(0).float()
            elif tf.dim() == 2 and tf.size(-1) == 768:
                text_b = tf.float()
            else:
                text_b = tf.reshape(tf, (-1, 768)).float()

            _head_in = int(self.MLPs[self.target_list[0]].layers[0].in_features)
            out_dict = {}
            loss_dict = {}
            total_loss = 0
            if _head_in == 768:
                for target_name in self.target_list:
                    out = self.MLPs[target_name](text_b)
                    out_dict[target_name] = out
                    y = _get_y_with_target(data, target_name)
                    if y is None:
                        continue
                    target = y.view((len(y), FLAGS.out_dim))
                    loss = self.loss_fucntion(out, target)
                    total_loss += loss
                    loss_dict[target_name] = loss
                return out_dict, total_loss, loss_dict

            if self.MLP1 is None:
                raise RuntimeError("LM 分支需要 MLP1，或 checkpoint 已将 768 维头折叠进 MLP")
            out_embed = self.MLP1(text_b)
            for target_name in self.target_list:
                out = self.MLPs[target_name](out_embed)
                out_dict[target_name] = out
                y = _get_y_with_target(data, target_name)
                if y is None:
                    continue
                target = y.view((len(y), FLAGS.out_dim))
                loss = self.loss_fucntion(out, target)
                total_loss += loss
                loss_dict[target_name] = loss
            return out_dict, total_loss, loss_dict

        out_embed = self.gnn_model(data)
        h_g = out_embed
        h_s = data.text_feat
        h_a = data.ast_feat

        if not getattr(FLAGS, "expert_rmse_silent_fusion", True):
            print(f"此时特征融合方案是：{ablation_mode}")

        if ablation_mode == "ast_text":
            h_g = torch.zeros_like(h_g)
        elif ablation_mode == "ast_cdfg":
            h_s = torch.zeros_like(h_s)
        elif ablation_mode == "cdfg_text":
            h_a = torch.zeros_like(h_a)

        if self.fusion_block is None:
            raise RuntimeError("fusion_block 未构建但进入融合前向，请检查 FLAGS")
        h_fuse = self.fusion_block(h_g=h_g, h_s=h_s, h_a=h_a)
        out_dict = {}
        loss_dict = {}
        total_loss = 0
        for target_name in self.target_list:
            out = self.MLPs[target_name](h_fuse)
            out_dict[target_name] = out
            y = _get_y_with_target(data, target_name)
            if y is None:
                continue
            target = y.view((len(y), FLAGS.out_dim))
            loss = self.loss_fucntion(out, target)
            total_loss += loss
            loss_dict[target_name] = loss
        return out_dict, total_loss, loss_dict
