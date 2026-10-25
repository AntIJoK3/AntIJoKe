from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FraudGraphSAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int = 256,
        out_channels: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
        self.convs.append(SAGEConv(hidden, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
