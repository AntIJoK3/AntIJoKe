from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import HTTPException
from neo4j.exceptions import ServiceUnavailable

from crypo_side.api.alerts import build_alert_if_needed, fetch_alert_evidence, gang_hint_from_evidence
from crypo_side.api.infer import (
    fetch_subgraph_tensors,
    heuristic_fraud_score,
    run_graphsage_forward,
    write_gnn_prob,
)
from crypo_side.collection.enrich import enrich_url_async
from crypo_side.db import make_driver
from crypo_side.graph.importer import merge_website_graph, set_text_embedding
from crypo_side.ml.embeddings import embed_texts


def _ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


def _merge_website(data: Any, label: str | None) -> None:
    with make_driver().session() as session:
        merge_website_graph(session, data, label)


def _embed_text(data: Any) -> list[float]:
    return embed_texts([data.visible_text or data.domain])[0]


def _set_embedding(url: str, vec: list[float]) -> None:
    with make_driver().session() as session:
        set_text_embedding(session, url, vec)


async def run_predict_async(
    url: str,
    label: str | None = None,
    *,
    profile: bool = False,
    enrich_timeout: float = 5.0,
    dns_lifetime: float = 2.0,
    whois_timeout: float = 2.0,
    cert_timeout: float = 2.0,
) -> tuple[dict[str, Any], dict[str, float] | None]:
    """异步 /predict 核心流程：丰富化并发 I/O，MERGE 与 Sentence-BERT 并行。"""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    t = t0
    data = await enrich_url_async(
        url,
        timeout=enrich_timeout,
        dns_lifetime=dns_lifetime,
        whois_timeout=whois_timeout,
        cert_timeout=cert_timeout,
    )
    t1 = time.perf_counter()
    if profile:
        timings["enrich"] = _ms(t, t1)

    try:
        merge_ms = 0.0
        embed_ms = 0.0
        vec: list[float] = []

        async def _merge_task() -> None:
            nonlocal merge_ms
            t_m = time.perf_counter()
            await asyncio.to_thread(_merge_website, data, label)
            merge_ms = _ms(t_m, time.perf_counter())

        async def _embed_task() -> None:
            nonlocal vec, embed_ms
            t_e = time.perf_counter()
            vec = await asyncio.to_thread(_embed_text, data)
            embed_ms = _ms(t_e, time.perf_counter())

        t = t1
        await asyncio.gather(_merge_task(), _embed_task())
        t2 = time.perf_counter()
        if profile:
            timings["merge"] = merge_ms
            timings["embed"] = embed_ms

        t = t2
        await asyncio.to_thread(_set_embedding, data.url, vec)
        t3 = time.perf_counter()
        if profile:
            timings["embed"] = embed_ms + _ms(t, t3)

        gang = None
        mode = "gnn"
        t = t3
        try:
            tensors = await asyncio.to_thread(fetch_subgraph_tensors, data.url, vec)
            t4 = time.perf_counter()
            if profile:
                timings["cypher"] = _ms(t, t4)

            t = t4
            prob = await asyncio.to_thread(run_graphsage_forward, tensors)
            t5 = time.perf_counter()
            if profile:
                timings["gnn"] = _ms(t, t5)
        except FileNotFoundError:
            t = t3
            prob = await asyncio.to_thread(heuristic_fraud_score, data.url)
            t5 = time.perf_counter()
            if profile:
                timings["cypher"] = _ms(t, t5)
                timings["gnn"] = 0.0
            mode = "heuristic"

        t = t5
        await asyncio.to_thread(write_gnn_prob, data.url, prob)
        evidence = await asyncio.to_thread(fetch_alert_evidence, data.url)
        gang = gang_hint_from_evidence(evidence)
        alert = await asyncio.to_thread(build_alert_if_needed, data.url, prob, mode)
        t6 = time.perf_counter()
        if profile:
            timings["other"] = _ms(t, t6)
            timings["e2e"] = _ms(t0, t6)

        payload = {
            "url": data.url,
            "fraud_probability": float(prob),
            "gang_hint": gang,
            "mode": mode,
            "alert": alert,
        }
        return payload, timings if profile else None

    except ServiceUnavailable as e:
        raise HTTPException(
            status_code=503,
            detail="无法连接 Neo4j，请确认数据库已启动且 .env 中 NEO4J_URI/密码正确。",
        ) from e


def run_predict(
    url: str,
    label: str | None = None,
    *,
    profile: bool = False,
) -> tuple[dict[str, Any], dict[str, float] | None]:
    """同步入口（脚本/测试）；API 请用 run_predict_async。"""
    coro = run_predict_async(url, label, profile=profile)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(
            asyncio.run,
            run_predict_async(url, label, profile=profile),
        ).result()
