"""表 6.6：在线检测各环节平均耗时（毫秒）。"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass

from crypo_side.api.infer import (
    fetch_subgraph_tensors,
    get_cached_model,
    run_graphsage_forward,
    write_gnn_prob,
)
from crypo_side.collection.enrich import EnrichmentResult, enrich_url
from crypo_side.db import make_driver
from crypo_side.graph.importer import merge_website_graph, set_text_embedding
from crypo_side.graph.rel_types import website_homog_edge_types
from crypo_side.ml.embeddings import embed_texts


@dataclass
class StageStats:
    name: str
    mean_ms: float
    std_ms: float
    share_pct: float


STAGE_KEYS = [
    ("enrich", "URL 丰富化（HTTP/DNS/WHOIS）"),
    ("merge", "图写入（MERGE）"),
    ("embed", "文本嵌入（Sentence-BERT）"),
    ("cypher", "1-hop 子图提取（Cypher）"),
    ("gnn", "GraphSAGE 推理"),
    ("other", "其他（序列化/网络）"),
    ("e2e", "端到端总耗时"),
]


def _ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


def _post_infer(url: str, prob: float) -> None:
    write_gnn_prob(url, prob)
    driver = make_driver()
    rel = website_homog_edge_types()
    with driver.session() as session:
        session.run(
            f"""
            MATCH (w:Website {{url:$url}})
            OPTIONAL MATCH (w)-[:{rel}]-(x:Website)
            WHERE x.label = 'fraud'
            RETURN w.louvain_community AS lc, count(DISTINCT x) AS xf
            """,
            url=url,
        ).consume()


def _sample_nodes(n: int) -> list[dict]:
    driver = make_driver()
    with driver.session() as session:
        rows = session.run(
            """
            MATCH (w:Website)
            WHERE w.url IS NOT NULL
            RETURN w.url AS url, w.domain AS domain, w.title AS title,
                   coalesce(w.content_hash, '') AS content_hash
            ORDER BY rand()
            LIMIT $n
            """,
            n=n,
        ).data()
    if not rows:
        raise SystemExit("Neo4j 中无 Website 节点，请先导入数据。")
    return rows


def _enrichment_from_row(row: dict) -> EnrichmentResult:
    return EnrichmentResult(
        url=row["url"],
        domain=row.get("domain") or row["url"],
        title=row.get("title") or "",
        visible_text=row.get("title") or row.get("domain") or "",
        ips=[],
        ns_names=[],
        registrant_emails=[],
        analytics_ids=[],
        cert_fingerprint=None,
        cert_subject_cn=None,
        content_hash=row.get("content_hash") or "",
        raw_error=None,
    )


def run_once(
    *,
    url: str,
    skip_enrich: bool,
    cached_row: dict | None,
    do_enrich: bool,
) -> dict[str, float]:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    if do_enrich:
        t = t0
        data = enrich_url(url)
        t1 = time.perf_counter()
        timings["enrich"] = _ms(t, t1)
    elif skip_enrich:
        assert cached_row is not None
        data = _enrichment_from_row(cached_row)
        timings["enrich"] = 0.0
        t1 = time.perf_counter()
    else:
        data = enrich_url(url)
        timings["enrich"] = _ms(t0, time.perf_counter())
        t1 = time.perf_counter()

    t = t1
    with make_driver().session() as session:
        merge_website_graph(session, data, None)
    t2 = time.perf_counter()
    timings["merge"] = _ms(t, t2)

    t = t2
    vec = embed_texts([data.visible_text or data.domain])[0]
    with make_driver().session() as session:
        set_text_embedding(session, data.url, vec)
    t3 = time.perf_counter()
    timings["embed"] = _ms(t, t3)

    t = t3
    tensors = fetch_subgraph_tensors(data.url, vec)
    t4 = time.perf_counter()
    timings["cypher"] = _ms(t, t4)

    t = t4
    prob = run_graphsage_forward(tensors)
    t5 = time.perf_counter()
    timings["gnn"] = _ms(t, t5)

    t = t5
    _post_infer(data.url, prob)
    _ = json.dumps({"url": data.url, "fraud_probability": prob, "mode": "gnn"})
    t6 = time.perf_counter()
    timings["other"] = _ms(t, t6)

    timings["e2e"] = _ms(t0, t6)
    return timings


def summarize(all_runs: list[dict[str, float]]) -> list[StageStats]:
    e2e_mean = statistics.mean(r["e2e"] for r in all_runs)
    out: list[StageStats] = []
    for key, label in STAGE_KEYS:
        vals = [r[key] for r in all_runs if key in r and r[key] == r[key]]  # skip NaN
        if not vals:
            vals = [0.0]
        mean_v = statistics.mean(vals)
        std_v = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        share = 100.0 if key == "e2e" else (mean_v / e2e_mean * 100.0 if e2e_mean > 0 else 0.0)
        out.append(StageStats(label, mean_v, std_v, share))
    return out


def _print_table(stats: list[StageStats], n: int) -> None:
    headers = ["环节", "平均耗时 (ms)", "标准差 (ms)", "占比"]
    rows = [
        [s.name, f"{s.mean_ms:.1f}", f"{s.std_ms:.1f}", f"{s.share_pct:.1f}%"]
        for s in stats
    ]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print("\n" + "=" * 72)
    print("表 6.6 在线检测各环节平均耗时（毫秒）")
    print("=" * 72)
    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in rows:
        print(fmt(row))
    print("=" * 72)
    print(f"有效样本 n={n}；占比以端到端均值 {stats[-1].mean_ms:.1f} ms 为分母。")


def main() -> None:
    p = argparse.ArgumentParser(description="在线检测各环节耗时基准")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--url-count", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument(
        "--skip-enrich",
        action="store_true",
        help="跳过 HTTP/DNS/WHOIS 丰富化（用库内节点属性），其余环节照常计时",
    )
    p.add_argument(
        "--enrich-samples",
        type=int,
        default=3,
        help="对 --enrich-urls 前 N 个做真实 enrich_url 计时",
    )
    p.add_argument(
        "--enrich-urls",
        nargs="*",
        default=[
            "https://www.wikipedia.org",
            "https://example.com",
            "https://www.cloudflare.com",
        ],
        help="真实 enrich 计时的 URL（默认用稳定可达站点）",
    )
    args = p.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    nodes = _sample_nodes(args.url_count)
    print(
        f"本地环节：{len(nodes)} URL × {args.rounds} 次"
        f"{'（跳过 enrich）' if args.skip_enrich else ''}；"
        f"HF 离线模式={os.environ['HF_HUB_OFFLINE']}\n"
    )

    for i in range(args.warmup):
        row = nodes[i % len(nodes)]
        try:
            run_once(
                url=row["url"],
                skip_enrich=True,
                cached_row=row,
                do_enrich=False,
            )
            print(f"  warmup {i + 1}/{args.warmup} ok")
        except Exception as e:  # noqa: BLE001
            print(f"  warmup {i + 1} 失败: {e}")

    embed_texts(["warmup"])
    get_cached_model()

    all_runs: list[dict[str, float]] = []
    for row in nodes:
        for _ in range(args.rounds):
            try:
                timings = run_once(
                    url=row["url"],
                    skip_enrich=args.skip_enrich,
                    cached_row=row,
                    do_enrich=False,
                )
                if args.skip_enrich:
                    timings["enrich"] = float("nan")
                all_runs.append(timings)
            except Exception as e:  # noqa: BLE001
                print(f"  跳过 {row['url']}: {e}")

    enrich_vals: list[float] = []
    if args.enrich_samples > 0 and args.enrich_urls:
        print(f"\n真实 URL 丰富化采样 n={len(args.enrich_urls[: args.enrich_samples])} …")
        for url in args.enrich_urls[: args.enrich_samples]:
            try:
                t0 = time.perf_counter()
                enrich_url(url)
                enrich_vals.append(_ms(t0, time.perf_counter()))
                print(f"  enrich {url[:50]} -> {enrich_vals[-1]:.1f} ms")
            except Exception as e:  # noqa: BLE001
                print(f"  enrich 失败 {url}: {e}")

    if not all_runs:
        raise SystemExit("无有效计时样本。")

    stats = summarize(all_runs)

    if enrich_vals:
        em = statistics.mean(enrich_vals)
        es = statistics.pstdev(enrich_vals) if len(enrich_vals) > 1 else 0.0
        pipeline_mean = statistics.mean(r["e2e"] for r in all_runs)
        total = pipeline_mean + em
        for s in stats:
            if s.name.startswith("URL 丰富化"):
                s.mean_ms = em
                s.std_ms = es
                s.share_pct = em / total * 100.0 if total > 0 else 0.0
            elif s.name == "端到端总耗时":
                s.mean_ms = total
                s.std_ms = (
                    statistics.pstdev([r["e2e"] + em for r in all_runs])
                    if len(all_runs) > 1
                    else es
                )
                s.share_pct = 100.0
            else:
                s.share_pct = s.mean_ms / total * 100.0 if total > 0 else 0.0

    _print_table(stats, len(all_runs))


if __name__ == "__main__":
    main()
