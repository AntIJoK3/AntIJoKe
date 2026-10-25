"""当 Neo4j GDS 插件不可用时，用 NetworkX + TruncatedSVD 近似图特征并写回 Website。"""
from __future__ import annotations


import networkx as nx
import numpy as np
from pathlib import Path

from typing import Any

from neo4j import Session
from sklearn.decomposition import TruncatedSVD

from crypo_side.graph.rel_types import website_homog_edge_types


def _load_website_graph(session: Session) -> tuple[list[str], list[tuple[str, str]]]:
    rows = session.run(
        """
        MATCH (w:Website)
        WHERE w.text_embedding IS NOT NULL
        RETURN w.url AS url
        """
    ).data()
    urls = [r["url"] for r in rows]
    rel = website_homog_edge_types()
    edge_rows = session.run(
        f"""
        MATCH (w1:Website)-[:{rel}]-(w2:Website)
        RETURN w1.url AS u1, w2.url AS u2
        """
    ).data()
    edges: list[tuple[str, str]] = []
    for r in edge_rows:
        u1, u2 = r["u1"], r["u2"]
        if u1 and u2 and u1 != u2:
            edges.append((u1, u2))
    return urls, edges


def compute_and_write_website_graph_features(session: Session) -> None:
    urls, edges = _load_website_graph(session)
    if not urls:
        raise ValueError("没有带 text_embedding 的 Website 节点")

    G = nx.Graph()
    for u in urls:
        G.add_node(u)
    G.add_edges_from(edges)

    pr = nx.pagerank(G, max_iter=100, tol=1e-4)
    # 500 节点量级可用精确介数；边极多时再改为采样
    btw = nx.betweenness_centrality(G, normalized=True) if G.number_of_nodes() <= 800 else {}

    comps = list(nx.connected_components(G))
    wcc: dict[str, int] = {}
    for idx, comp in enumerate(comps):
        for u in comp:
            wcc[u] = idx

    nodes_sorted = sorted(urls)
    idx = {u: i for i, u in enumerate(nodes_sorted)}
    n = len(nodes_sorted)
    adj = np.zeros((n, n), dtype=np.float32)
    for u, v in edges:
        if u in idx and v in idx:
            i, j = idx[u], idx[v]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    np.fill_diagonal(adj, 1.0)

    n_comp = min(128, max(2, n - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    emb = svd.fit_transform(adj)
    if n_comp < 128:
        pad = np.zeros((n, 128 - n_comp), dtype=np.float32)
        emb = np.hstack([emb, pad])
    else:
        emb = emb[:, :128]

    rows_out: list[dict[str, Any]] = []
    max_deg = max((G.degree(u) for u in nodes_sorted), default=1)
    for u in nodes_sorted:
        deg = G.degree(u) if u in G else 0
        pr_u = float(pr.get(u, 0.0))
        if pr_u == 0.0 and deg > 0:
            pr_u = (deg / max_deg) * 0.01 + 1e-6
        btw_u = float(btw.get(u, 0.0)) if btw else 0.0
        vec = emb[idx[u]].astype(float).tolist()
        rows_out.append(
            {
                "url": u,
                "pagerank": pr_u,
                "betweenness": btw_u,
                "wcc_id": int(wcc.get(u, 0)),
                "louvain_community": int(wcc.get(u, 0)),
                "graph_embedding": vec,
            }
        )

    for i in range(0, len(rows_out), 200):
        batch = rows_out[i : i + 200]
        session.run(
            """
            UNWIND $rows AS row
            MATCH (w:Website {url: row.url})
            SET w.pagerank = row.pagerank,
                w.betweenness = row.betweenness,
                w.wcc_id = row.wcc_id,
                w.louvain_community = row.louvain_community,
                w.graph_embedding = row.graph_embedding,
                w.feature_backend = 'networkx_svd'
            """,
            rows=batch,
        )


def _mark_gds_backend(session: Session) -> None:
    """GDS write 成功后标记来源（pageRank/betweenness 等已写回节点）。"""
    session.run(
        """
        MATCH (w:Website)
        WHERE w.pagerank IS NOT NULL AND w.graph_embedding IS NOT NULL
        SET w.feature_backend = 'gds'
        """
    )


def try_run_gds_then_concat(session: Session, base: Path, gds_projection: str = "minimal") -> None:
    from crypo_side.graph.cypher_runner import run_cypher_file

    proj_name = "03_gds_project_full.cypher" if gds_projection == "full" else "03_gds_project.cypher"
    try:
        run_cypher_file(session, base / proj_name)
        run_cypher_file(session, base / "04_gds_features.cypher")
        _mark_gds_backend(session)
        print("已使用 Neo4j GDS 写入图特征")
    except Exception as e:  # noqa: BLE001
        low = str(e).lower()
        if "gds" in low or "procedure" in low or "not found" in low:
            print("未检测到 GDS 或投影失败，回退 NetworkX/Sklearn 特征:", e)
            compute_and_write_website_graph_features(session)
        else:
            raise
    run_cypher_file(session, base / "06_feat_vector_concat.cypher")
