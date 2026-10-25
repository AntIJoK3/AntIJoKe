from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from crypo_side.config import get_settings
from crypo_side.db import make_driver
from crypo_side.graph.rel_types import website_homog_edge_types


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_alert_evidence(url: str) -> dict[str, Any]:
    """收集预警关联证据：共享 IP/NS、Louvain 社区及社区内欺诈节点数等。"""
    rel = website_homog_edge_types()
    with make_driver().session() as session:
        row = session.run(
            f"""
            MATCH (w:Website {{url: $url}})
            OPTIONAL MATCH (w)-[:HAS_IP]->(ip:IP)<-[:HAS_IP]-(ow:Website)
            WHERE ow <> w
            WITH w, collect(DISTINCT ip.address) AS shared_ips
            OPTIONAL MATCH (w)-[:USES_NS]->(ns:NameServer)<-[:USES_NS]-(on:Website)
            WHERE on <> w
            WITH w, shared_ips, collect(DISTINCT ns.ns_name) AS shared_ns
            WITH w, shared_ips, shared_ns, w.louvain_community AS lc
            OPTIONAL MATCH (cf:Website)
            WHERE cf.louvain_community = lc AND cf.label = 'fraud' AND cf <> w
            WITH w, shared_ips, shared_ns, lc, count(DISTINCT cf) AS community_fraud_count
            OPTIONAL MATCH (w)-[:{rel}]-(fn:Website)
            WHERE fn.label = 'fraud' AND fn <> w
            RETURN shared_ips AS shared_ips,
                   shared_ns AS shared_ns,
                   lc AS louvain_community,
                   community_fraud_count AS community_fraud_count,
                   count(DISTINCT fn) AS fraud_neighbor_count
            """,
            url=url,
        ).single()

    if row is None:
        return {
            "shared_ips": [],
            "shared_ns": [],
            "louvain_community": None,
            "community_fraud_count": 0,
            "fraud_neighbor_count": 0,
        }

    lc = row.get("louvain_community")
    return {
        "shared_ips": [str(x) for x in (row.get("shared_ips") or []) if x],
        "shared_ns": [str(x) for x in (row.get("shared_ns") or []) if x],
        "louvain_community": int(lc) if lc is not None else None,
        "community_fraud_count": int(row.get("community_fraud_count") or 0),
        "fraud_neighbor_count": int(row.get("fraud_neighbor_count") or 0),
    }


def gang_hint_from_evidence(evidence: dict[str, Any]) -> str | None:
    lc = evidence.get("louvain_community")
    if lc is None:
        return None
    return (
        f"louvain:{lc},fraud_neighbors:{int(evidence.get('fraud_neighbor_count') or 0)}"
    )


def persist_alert(
    url: str,
    fraud_probability: float,
    mode: str,
    evidence: dict[str, Any],
    *,
    threshold: float,
    triggered_at: str | None = None,
) -> dict[str, Any]:
    """将预警写入 Neo4j（:Alert 节点）。"""
    alert_id = str(uuid.uuid4())
    ts = triggered_at or _utc_now_iso()
    lc = evidence.get("louvain_community")

    with make_driver().session() as session:
        session.run(
            """
            MATCH (w:Website {url: $url})
            CREATE (a:Alert {
                alert_id: $alert_id,
                url: $url,
                fraud_probability: $prob,
                triggered_at: datetime($ts),
                mode: $mode,
                threshold: $threshold,
                shared_ips: $shared_ips,
                shared_ns: $shared_ns,
                louvain_community: $lc,
                community_fraud_count: $cfc,
                fraud_neighbor_count: $fnc
            })
            CREATE (a)-[:FOR_WEBSITE]->(w)
            """,
            url=url,
            alert_id=alert_id,
            prob=float(fraud_probability),
            ts=ts,
            mode=mode,
            threshold=float(threshold),
            shared_ips=evidence.get("shared_ips") or [],
            shared_ns=evidence.get("shared_ns") or [],
            lc=lc,
            cfc=int(evidence.get("community_fraud_count") or 0),
            fnc=int(evidence.get("fraud_neighbor_count") or 0),
        ).consume()

    return alert_record_dict(
        alert_id=alert_id,
        url=url,
        fraud_probability=fraud_probability,
        mode=mode,
        threshold=threshold,
        triggered_at=ts,
        evidence=evidence,
    )


def alert_record_dict(
    *,
    alert_id: str,
    url: str,
    fraud_probability: float,
    mode: str,
    threshold: float,
    triggered_at: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "alert_id": alert_id,
        "url": url,
        "fraud_probability": float(fraud_probability),
        "threshold": float(threshold),
        "triggered_at": triggered_at,
        "mode": mode,
        "evidence": evidence,
    }


def build_alert_if_needed(
    url: str,
    fraud_probability: float,
    mode: str,
    *,
    threshold: float | None = None,
    persist: bool = True,
) -> dict[str, Any] | None:
    """概率超过阈值时生成预警记录（可选持久化）。"""
    thr = threshold if threshold is not None else get_settings().fraud_alert_threshold
    if float(fraud_probability) < thr:
        return None

    evidence = fetch_alert_evidence(url)
    record = alert_record_dict(
        alert_id=str(uuid.uuid4()),
        url=url,
        fraud_probability=fraud_probability,
        mode=mode,
        threshold=thr,
        triggered_at=_utc_now_iso(),
        evidence=evidence,
    )
    if persist:
        saved = persist_alert(
            url,
            fraud_probability,
            mode,
            evidence,
            threshold=thr,
            triggered_at=record["triggered_at"],
        )
        return saved
    return record


def list_alerts(
    *,
    limit: int = 50,
    min_probability: float | None = None,
    since_iso: str | None = None,
) -> list[dict[str, Any]]:
    """查询历史预警记录。"""
    with make_driver().session() as session:
        rows = session.run(
            """
            MATCH (a:Alert)
            WHERE ($min_prob IS NULL OR a.fraud_probability >= $min_prob)
              AND ($since IS NULL OR a.triggered_at >= datetime($since))
            RETURN a
            ORDER BY a.triggered_at DESC
            LIMIT $limit
            """,
            min_prob=min_probability,
            since=since_iso,
            limit=max(1, min(limit, 500)),
        ).data()

    out: list[dict[str, Any]] = []
    for row in rows:
        a = dict(row["a"])
        triggered = a.get("triggered_at")
        if hasattr(triggered, "isoformat"):
            ts = triggered.isoformat()
        elif triggered is not None:
            ts = str(triggered)
        else:
            ts = ""
        evidence = {
            "shared_ips": list(a.get("shared_ips") or []),
            "shared_ns": list(a.get("shared_ns") or []),
            "louvain_community": a.get("louvain_community"),
            "community_fraud_count": int(a.get("community_fraud_count") or 0),
            "fraud_neighbor_count": int(a.get("fraud_neighbor_count") or 0),
        }
        lc = evidence["louvain_community"]
        if lc is not None:
            evidence["louvain_community"] = int(lc)
        out.append(
            alert_record_dict(
                alert_id=str(a.get("alert_id") or ""),
                url=str(a.get("url") or ""),
                fraud_probability=float(a.get("fraud_probability") or 0),
                mode=str(a.get("mode") or ""),
                threshold=float(a.get("threshold") or get_settings().fraud_alert_threshold),
                triggered_at=ts,
                evidence=evidence,
            )
        )
    return out
