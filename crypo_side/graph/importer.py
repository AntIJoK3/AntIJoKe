from __future__ import annotations

from typing import Any

from neo4j import Session

from crypo_side.collection.enrich import EnrichmentResult


def merge_website_graph(session: Session, data: EnrichmentResult, label: str | None) -> dict[str, Any]:
    """将一次采集结果写入 Neo4j（MERGE 节点与关系）。"""
    params: dict[str, Any] = {
        "url": data.url,
        "domain": data.domain,
        "title": data.title,
        "content_hash": data.content_hash,
        "label": label,
    }
    cypher = """
    MERGE (w:Website {url: $url})
    SET w.domain = $domain,
        w.title = $title,
        w.content_hash = $content_hash
    """
    if label is not None:
        cypher += ", w.label = $label"
    session.run(cypher + "\nRETURN id(w) AS nid", params)

    for ip in data.ips:
        session.run(
            """
            MATCH (w:Website {url: $url})
            MERGE (i:IP {address: $ip})
            MERGE (w)-[:HAS_IP]->(i)
            """,
            {"url": data.url, "ip": ip},
        )

    for ns in data.ns_names:
        session.run(
            """
            MATCH (w:Website {url: $url})
            MERGE (n:NameServer {ns_name: $ns})
            MERGE (w)-[:USES_NS]->(n)
            """,
            {"url": data.url, "ns": ns},
        )

    for email in data.registrant_emails:
        session.run(
            """
            MATCH (w:Website {url: $url})
            MERGE (e:RegistrantEmail {email: $email})
            MERGE (w)-[:REGISTERED_BY]->(e)
            """,
            {"url": data.url, "email": email},
        )

    for kind, tid in data.analytics_ids:
        session.run(
            """
            MATCH (w:Website {url: $url})
            MERGE (a:AnalyticsID {tracker_id: $tid})
            ON CREATE SET a.type = $kind
            MERGE (w)-[:HAS_ANALYTICS]->(a)
            """,
            {"url": data.url, "tid": tid, "kind": kind},
        )

    if data.cert_fingerprint:
        session.run(
            """
            MATCH (w:Website {url: $url})
            MERGE (c:SSLCert {fingerprint: $fp})
            SET c.subject_cn = coalesce($cn, c.subject_cn)
            MERGE (w)-[:HAS_CERT]->(c)
            """,
            {"url": data.url, "fp": data.cert_fingerprint, "cn": data.cert_subject_cn},
        )

    return {"url": data.url, "domain": data.domain}


def set_text_embedding(session: Session, url: str, vector: list[float]) -> None:
    session.run(
        """
        MATCH (w:Website {url: $url})
        SET w.text_embedding = $vec
        """,
        {"url": url, "vec": vector},
    )
