from __future__ import annotations

from typing import Any, NamedTuple

import torch
import torch.nn.functional as F

from crypo_side.config import get_settings
from crypo_side.db import make_driver
from crypo_side.graph.rel_types import website_homog_edge_types
from crypo_side.ml.graphsage import FraudGraphSAGE


def _load_model(device: torch.device) -> tuple[FraudGraphSAGE, dict[str, Any]]:
    path = get_settings().gnn_checkpoint
    if not path.exists():
        raise FileNotFoundError(str(path))
    ckpt = torch.load(path, map_location=device)
    h = ckpt.get("hparams", {})
    model = FraudGraphSAGE(
        int(ckpt["in_channels"]),
        hidden=int(h.get("hidden", 256)),
        out_channels=2,
        num_layers=int(h.get("num_layers", 2)),
        dropout=float(h.get("dropout", 0.3)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


_model_cache: tuple[FraudGraphSAGE, dict[str, Any]] | None = None


def get_cached_model(device: torch.device | None = None) -> tuple[FraudGraphSAGE, dict[str, Any]]:
    """进程内缓存 GraphSAGE 权重，避免在线推理重复 load。"""
    global _model_cache
    if device is None:
        device = torch.device("cpu")
    if _model_cache is None:
        _model_cache = _load_model(device)
    return _model_cache


class SubgraphTensors(NamedTuple):
    x: torch.Tensor
    edge_index: torch.Tensor
    center: int


def fetch_subgraph_tensors(url: str, text_embedding: list[float]) -> SubgraphTensors:
    """1-hop 子图 Cypher 查询 + 特征组装（不含 GNN 前向）。"""
    device = torch.device("cpu")
    model, _ = get_cached_model(device)
    dim = model.convs[0].in_channels
    if 130 + len(text_embedding) != dim:
        raise ValueError(f"嵌入维度 {len(text_embedding)} 与模型输入 {dim} 不匹配")

    driver = make_driver()
    with driver.session() as session:
        row = session.run(
            "MATCH (w:Website {url:$url}) RETURN w.url AS u",
            url=url,
        ).single()
        if row is None:
            raise ValueError("节点尚未写入图谱")

        nbr = session.run(
            f"""
            MATCH (w:Website {{url:$url}})
            OPTIONAL MATCH (w)-[:{website_homog_edge_types()}]-(n:Website)
            RETURN collect(DISTINCT n.url) AS nbrs
            """,
            url=url,
        ).single()
        raw_nbrs = nbr["nbrs"] if nbr and nbr["nbrs"] else []
        nbr_urls = [u for u in raw_nbrs if u]
        node_urls = list(dict.fromkeys([url] + nbr_urls))

        feats: dict[str, list[float]] = {}
        for u in node_urls:
            rec = session.run(
                """
                MATCH (w:Website {url: $u})
                RETURN w.feat_vector AS fv, w.text_embedding AS te, w.url AS uu
                """,
                u=u,
            ).single()
            fv = rec["fv"]
            if fv is None:
                te = rec["te"]
                if te is None and rec["uu"] == url:
                    te = text_embedding
                if te is None:
                    te = [0.0] * (dim - 130)
                fv = cold_start_feat_vector(te)
            feats[u] = list(fv)

        rec_edges = session.run(
            f"""
            UNWIND $urls AS u1
            MATCH (a:Website {{url: u1}})
            MATCH (a)-[:{website_homog_edge_types()}]-(b:Website)
            WHERE b.url IN $urls AND a.url < b.url
            RETURN collect(DISTINCT [a.url, b.url]) AS pairs
            """,
            urls=node_urls,
        ).single()
        raw_pairs = rec_edges["pairs"] if rec_edges and rec_edges["pairs"] else []
        pairs: list[tuple[str, str]] = []
        for item in raw_pairs:
            if item and len(item) == 2:
                pairs.append((str(item[0]), str(item[1])))

        idx = {u: i for i, u in enumerate(node_urls)}
        x = torch.tensor([feats[u] for u in node_urls], dtype=torch.float32, device=device)
        if not pairs:
            edge_index = torch.tensor([[idx[url]], [idx[url]]], dtype=torch.long, device=device)
        else:
            ei: list[list[int]] = []
            for a, b in pairs:
                ia, ib = idx[a], idx[b]
                ei.append([ia, ib])
                ei.append([ib, ia])
            edge_index = torch.tensor(ei, dtype=torch.long, device=device).t().contiguous()

        center = idx[url]

    return SubgraphTensors(x=x, edge_index=edge_index, center=center)


def run_graphsage_forward(tensors: SubgraphTensors) -> float:
    """GraphSAGE 前向推理（不含 Cypher）。"""
    device = torch.device("cpu")
    model, _ = get_cached_model(device)
    with torch.no_grad():
        logits = model(tensors.x, tensors.edge_index)
        prob = F.softmax(logits, dim=1)[tensors.center, 1].item()
    return float(prob)


def cold_start_feat_vector(text_embedding: list[float]) -> list[float]:
    """新节点尚未跑 GDS 时，用零填充结构维，保持与训练时 feat_vector 维度一致。"""
    return [0.0, 0.0] + [0.0] * 128 + list(text_embedding)


def infer_subgraph_for_url(url: str, text_embedding: list[float]) -> float:
    """对单 URL 与其一阶 Website 邻居构建子图并输出 fraud 类概率。"""
    tensors = fetch_subgraph_tensors(url, text_embedding)
    return run_graphsage_forward(tensors)


def write_gnn_prob(url: str, prob: float) -> None:
    driver = make_driver()
    with driver.session() as session:
        session.run(
            """
            MATCH (w:Website {url:$url})
            SET w.gnn_fraud_prob = $p
            """,
            url=url,
            p=prob,
        )


def heuristic_fraud_score(url: str) -> float:
    """无模型权重时：按与已知 fraud 站点的共享关联比例给出粗分。"""
    driver = make_driver()
    with driver.session() as session:
        rel = website_homog_edge_types()
        row = session.run(
            f"""
            MATCH (w:Website {{url:$url}})
            OPTIONAL MATCH (w)-[:{rel}]-(f:Website)
            WHERE f.label = 'fraud' AND f <> w
            WITH w, count(DISTINCT f) AS c
            OPTIONAL MATCH (w)-[:{rel}]-(n:Website)
            WITH c, count(DISTINCT n) AS tot
            RETURN c, tot
            """,
            url=url,
        ).single()
        if not row or row["tot"] == 0:
            return 0.0
        return min(1.0, float(row["c"]) / max(1, int(row["tot"])))


def _strip_large_props(props: dict) -> dict:
    out = {}
    for k, v in props.items():
        if k in ("text_embedding", "feat_vector", "graph_embedding"):
            continue
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v[:8]):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def export_subgraph(
    *,
    domain: str | None = None,
    url: str | None = None,
    center_id: str | None = None,
    max_edges: int = 120,
) -> dict[str, Any]:
    """Export 1-hop neighborhood around a Website (by domain/url) or any node (by elementId)."""
    if not any([domain, url, center_id]):
        raise ValueError("需提供 domain、url 或 center_id 之一")

    driver = make_driver()
    with driver.session() as session:
        if center_id:
            rows = session.run(
                """
                MATCH (center)
                WHERE elementId(center) = $cid
                WITH center LIMIT 1
                MATCH (center)-[r]-(neighbor)
                WITH center, neighbor, type(r) AS rt
                LIMIT $lim
                RETURN elementId(center) AS wid, labels(center) AS lw, properties(center) AS pw,
                       elementId(neighbor) AS mid, labels(neighbor) AS lm, properties(neighbor) AS pm,
                       rt
                """,
                cid=center_id,
                lim=max_edges,
            ).data()
        elif url:
            rows = session.run(
                """
                MATCH (center:Website {url: $url})
                WITH center LIMIT 1
                MATCH (center)-[r]-(neighbor)
                WITH center, neighbor, type(r) AS rt
                LIMIT $lim
                RETURN elementId(center) AS wid, labels(center) AS lw, properties(center) AS pw,
                       elementId(neighbor) AS mid, labels(neighbor) AS lm, properties(neighbor) AS pm,
                       rt
                """,
                url=url,
                lim=max_edges,
            ).data()
        else:
            rows = session.run(
                """
                MATCH (w:Website)
                WHERE w.domain = $d OR w.url ENDS WITH $d OR w.url CONTAINS $d
                WITH w ORDER BY CASE WHEN w.url STARTS WITH 'benchscale:' THEN 1 ELSE 0 END,
                              coalesce(w.gnn_fraud_prob, 0) DESC,
                              size(w.url) ASC,
                              w.domain ASC
                LIMIT 1
                MATCH (w)-[r]-(neighbor)
                WITH w AS center, neighbor, type(r) AS rt
                LIMIT $lim
                RETURN elementId(center) AS wid, labels(center) AS lw, properties(center) AS pw,
                       elementId(neighbor) AS mid, labels(neighbor) AS lm, properties(neighbor) AS pm,
                       rt
                """,
                d=domain,
                lim=max_edges,
            ).data()

        if not rows:
            return {
                "nodes": [],
                "edges": [],
                "center_id": center_id,
                "query": {"domain": domain, "url": url, "center_id": center_id},
            }

        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        center_nid = str(rows[0]["wid"])

        def add_node(nid: str, labels: list, props: dict) -> None:
            if nid in nodes:
                return
            nodes[nid] = {
                "id": nid,
                "labels": labels,
                "properties": _strip_large_props(dict(props)),
            }

        for row in rows:
            wid = str(row["wid"])
            mid = str(row["mid"])
            rt = row["rt"]
            add_node(wid, row["lw"], row["pw"])
            add_node(mid, row["lm"], row["pm"])
            edge_key = tuple(sorted((wid, mid))) + (rt,)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edges.append({"source": wid, "target": mid, "type": rt})

        center_node = nodes.get(center_nid, {})
        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            "center_id": center_nid,
            "center_labels": center_node.get("labels", []),
            "query": {"domain": domain, "url": url, "center_id": center_id or center_nid},
        }
