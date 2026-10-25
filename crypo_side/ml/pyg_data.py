from __future__ import annotations

import torch
from torch_geometric.data import Data

from crypo_side.graph.rel_types import website_homog_edge_types

# feat_vector = [pagerank, betweenness] + graph_embedding(128) + text_embedding(384)
GRAPH_EMBEDDING_DIM = 128
STRUCT_FEATURE_DIM = 2 + GRAPH_EMBEDDING_DIM


def select_feature_columns(x: torch.Tensor, mode: str) -> torch.Tensor:
    """按特征消融模式切片节点特征矩阵。"""
    if mode == "fusion":
        return x
    if mode == "structure":
        return x[:, :STRUCT_FEATURE_DIM]
    if mode == "content":
        return x[:, STRUCT_FEATURE_DIM:]
    raise ValueError(f"未知 feature mode: {mode!r}，可选 fusion / structure / content")


def build_pyg_data(session) -> tuple[Data, dict[str, int]]:
    nodes = session.run(
        """
        MATCH (w:Website)
        WHERE w.feat_vector IS NOT NULL AND w.label IS NOT NULL
        RETURN w.url AS url, w.feat_vector AS feat, w.label AS label
        ORDER BY url
        """
    ).data()
    if not nodes:
        raise ValueError("没有带 feat_vector 与 label 的 Website 节点，请先完成采集、嵌入与 GDS 特征。")

    id_map = {n["url"]: i for i, n in enumerate(nodes)}
    x = torch.tensor([n["feat"] for n in nodes], dtype=torch.float32)
    y = torch.tensor([1 if n["label"] == "fraud" else 0 for n in nodes], dtype=torch.long)

    urls = list(id_map.keys())
    rel = website_homog_edge_types()
    edges = session.run(
        f"""
        MATCH (w1:Website)-[:{rel}]-(w2:Website)
        WHERE w1.url IN $urls AND w2.url IN $urls AND w1.url < w2.url
        RETURN DISTINCT w1.url AS source, w2.url AS target
        """,
        urls=urls,
    ).data()

    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        src = [id_map[e["source"]] for e in edges]
        tgt = [id_map[e["target"]] for e in edges]
        edge_index = torch.tensor([src + tgt, tgt + src], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data, id_map
