from __future__ import annotations

from typing import Any

from neo4j import Session


def merge_website_batch(session: Session, rows: list[dict[str, Any]]) -> int:
    """批量写入 Website、IP 与 HAS_IP（来自 DNS 已解析数据集）。"""
    if not rows:
        return 0
    session.run(
        """
        UNWIND $rows AS row
        MERGE (w:Website {url: row.url})
        SET w.domain = row.domain,
            w.label = row.label,
            w.source_label = row.source_label,
            w.dns_ttl = row.ttl,
            w.first_seen = row.first_seen,
            w.dataset = 'urls_with_dns_live_mixed'
        """,
        rows=rows,
    )
    edges: list[dict[str, Any]] = []
    for row in rows:
        for ip in row.get("ips") or []:
            edges.append({"url": row["url"], "ip": ip, "first_seen": row.get("first_seen")})
    if edges:
        session.run(
            """
            UNWIND $edges AS e
            MATCH (w:Website {url: e.url})
            MERGE (i:IP {address: e.ip})
            MERGE (w)-[r:HAS_IP]->(i)
            ON CREATE SET r.first_seen = e.first_seen, r.last_seen = e.first_seen
            ON MATCH SET r.last_seen = coalesce(e.first_seen, r.last_seen)
            """,
            edges=edges,
        )
    return len(rows)


def set_text_embeddings_batch(session: Session, pairs: list[dict[str, Any]]) -> None:
    """pairs: [{url, vec}, ...]"""
    if not pairs:
        return
    session.run(
        """
        UNWIND $pairs AS p
        MATCH (w:Website {url: p.url})
        SET w.text_embedding = p.vec
        """,
        pairs=pairs,
    )
