"""
ProgSG graph encoder adapter for MPM-Innovation comparative experiments.

Official implementation (full CDFG + CodeT5 + node-token interaction):
  https://github.com/ZongyueQin/ProgSG

This module reimplements the **GNN branch** of ProgSG (TransformerConv stack,
JumpingKnowledge, global attention pooling) so it can plug into ``src/model.py::Net``
via ``--comparative_if True --comparative_model progsg``.

For end-to-end reproduction of published ProgSG / Table-IV PROGSG numbers, use the
external repo above (``python main.py``, pretrained weights under ``src/logs/``).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, JumpingKnowledge, TransformerConv
from torch_geometric.nn import global_add_pool
from torch.nn import Linear, ReLU, Sequential

from src.config import FLAGS
from src.nn_att import MyGlobalAttention


def _activation_fn():
    act = getattr(FLAGS, "progsg_activation", getattr(FLAGS, "activation", "elu"))
    if act == "relu":
        return F.relu
    if act == "elu":
        return F.elu
    raise NotImplementedError(f"Unsupported activation: {act}")


class ProgSGNet(torch.nn.Module):
    """
    ProgSG-style CDFG encoder (graph modality only).

    Aligns with ProgSG ``src/model.py`` GNN path:
      - ``gnn_type='transformer'`` (default in ProgSG config)
      - ``num_layers=8``, ``jkn_mode='max'``, node-level attention pooling
    """

    def __init__(
        self,
        in_channels: int,
        edge_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        D: Optional[int] = None,
    ):
        super().__init__()
        self.D = int(D if D is not None else FLAGS.D)
        self.num_layers = int(
            num_layers
            if num_layers is not None
            else getattr(FLAGS, "progsg_num_layers", 8)
        )
        self.edge_dim = int(
            edge_dim
            if edge_dim is not None
            else getattr(FLAGS, "progsg_edge_dim", 7)
        )

        gnn_type = getattr(FLAGS, "progsg_gnn_type", "transformer")
        if gnn_type == "gat":
            conv_class = GATConv
        elif gnn_type == "gcn":
            conv_class = GCNConv
        elif gnn_type == "transformer":
            conv_class = TransformerConv
        else:
            raise NotImplementedError(f"Unsupported progsg_gnn_type={gnn_type}")

        self._is_transformer = conv_class is TransformerConv
        encode_edge = getattr(FLAGS, "progsg_encode_edge", True)
        self._use_edge = bool(encode_edge and self._is_transformer)

        if self._use_edge:
            self.conv_first = conv_class(in_channels, self.D, edge_dim=self.edge_dim)
        else:
            self.conv_first = conv_class(in_channels, self.D)

        self.conv_layers = nn.ModuleList()
        for _ in range(max(0, self.num_layers - 1)):
            if self._use_edge:
                self.conv_layers.append(conv_class(self.D, self.D, edge_dim=self.edge_dim))
            else:
                self.conv_layers.append(conv_class(self.D, self.D))

        jkn_mode = getattr(FLAGS, "progsg_jkn_mode", "max")
        self.jkn_enable = bool(getattr(FLAGS, "progsg_jkn_enable", True))
        self.jkn = JumpingKnowledge(jkn_mode, channels=self.D, num_layers=2)

        self.node_attention = bool(getattr(FLAGS, "progsg_node_attention", True))
        if self.node_attention:
            self.gate_nn = Sequential(Linear(self.D, self.D), ReLU(), Linear(self.D, 1))
            self.glob = MyGlobalAttention(self.gate_nn, None)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)
        activation = _activation_fn()

        outs = []
        if self._use_edge and edge_attr is not None:
            out = activation(self.conv_first(x, edge_index, edge_attr=edge_attr))
        else:
            out = activation(self.conv_first(x, edge_index))
        outs.append(out)

        for i, conv in enumerate(self.conv_layers):
            if self._use_edge and edge_attr is not None:
                out = conv(out, edge_index, edge_attr=edge_attr)
            else:
                out = conv(out, edge_index)
            if i != len(self.conv_layers) - 1:
                out = activation(out)
            outs.append(out)

        if self.jkn_enable:
            out = self.jkn(outs)

        if self.node_attention:
            out, _ = self.glob(out, batch)
        else:
            out = global_add_pool(out, batch)

        return out


# Alias used by ``src/model.py`` (same pattern as ``GNN_DSE.Net``).
Net = ProgSGNet


def try_load_external_progsg_model(checkpoint_path: Optional[str] = None):
    """
    Optional helper: load weights from a cloned ProgSG repo.

    Set FLAGS.progsg_external_root to the ProgSG project root (contains ``src/``).
    Set FLAGS.progsg_checkpoint to ``.pth`` path (or use default under ``src/logs/``).

    Returns the external ``Net`` instance, or None if the repo is unavailable.
    """
    root = getattr(FLAGS, "progsg_external_root", None) or os.environ.get("PROGSG_ROOT")
    if not root or not os.path.isdir(root):
        return None

    src_dir = os.path.join(root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    try:
        from model import Net as ExternalProgSGNet  # type: ignore
    except Exception:
        return None

    ckpt = checkpoint_path or getattr(FLAGS, "progsg_checkpoint", None)
    if ckpt is None:
        return None

    model = ExternalProgSGNet()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model
