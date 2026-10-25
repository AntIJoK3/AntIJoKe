"""表 6.6：POST /predict 端到端耗时（单线程顺序，500 URL 抽样）。"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient

from crypo_side.api.main import app
from crypo_side.api.infer import get_cached_model
from crypo_side.db import make_driver
from crypo_side.ml.embeddings import embed_texts
from crypo_side.ml.train import stratified_train_val_test_masks
from crypo_side.ml.pyg_data import build_pyg_data


STAGE_ROWS = [
    ("enrich", "URL 丰富化（HTTP/DNS/WHOIS）"),
    ("merge", "图写入（MERGE）"),
    ("embed", "文本嵌入（Sentence-BERT）"),
    ("cypher", "1-hop 子图提取（Cypher）"),
    ("gnn", "GraphSAGE 推理"),
    ("other", "其他（序列化/网络）"),
    ("e2e", "端到端总耗时"),
]


@dataclass
class StageStats:
    name: str
    mean_ms: float
    std_ms: float
    share_pct: float


def sample_test_urls(n_per_class: int, seed: int, *, active_only: bool = False) -> list[str]:
    """从 Neo4j 随机抽取 fraud/normal 各 n_per_class 条 URL。"""
    active_filter = (
        "AND w.title IS NOT NULL AND trim(w.title) <> '' "
        "AND w.content_hash IS NOT NULL AND trim(w.content_hash) <> ''"
        if active_only
        else ""
    )
    driver = make_driver()
    with driver.session() as session:
        fraud = session.run(
            f"""
            MATCH (w:Website)
            WHERE w.label = 'fraud' AND w.url IS NOT NULL {active_filter}
            RETURN w.url AS url
            ORDER BY rand()
            LIMIT $n
            """,
            n=n_per_class,
        ).data()
        normal = session.run(
            f"""
            MATCH (w:Website)
            WHERE w.label = 'normal' AND w.url IS NOT NULL {active_filter}
            RETURN w.url AS url
            ORDER BY rand()
            LIMIT $n
            """,
            n=n_per_class,
        ).data()
    urls = [r["url"] for r in fraud] + [r["url"] for r in normal]
    rng = random.Random(seed)
    rng.shuffle(urls)
    return urls


def sample_test_urls_from_split(
    n_per_class: int, seed: int, *, active_only: bool = False
) -> list[str]:
    """优先从 stratified 测试集划分中抽取（与训练脚本一致）。"""
    driver = make_driver()
    with driver.session() as session:
        data, _ = build_pyg_data(session)
        urls_all = session.run(
            """
            MATCH (w:Website)
            WHERE w.feat_vector IS NOT NULL AND w.label IS NOT NULL
            RETURN w.url AS url, w.label AS label
            ORDER BY w.url
            """
        ).data()
    url_to_label = {r["url"]: r["label"] for r in urls_all}
    _, _, test_mask = stratified_train_val_test_masks(
        data.y, test_size=0.2, val_fraction_of_trainval=0.25, seed=seed
    )
    test_idx = test_mask.nonzero(as_tuple=False).view(-1).tolist()
    ordered_urls = [r["url"] for r in urls_all]
    test_urls = [ordered_urls[i] for i in test_idx]
    fraud = [u for u in test_urls if url_to_label.get(u) == "fraud"]
    normal = [u for u in test_urls if url_to_label.get(u) == "normal"]
    rng = random.Random(seed + 1)
    rng.shuffle(fraud)
    rng.shuffle(normal)
    picked = fraud[:n_per_class] + normal[:n_per_class]
    if len(picked) < n_per_class * 2:
        print(
            f"警告: 测试集仅 {len(fraud)} fraud / {len(normal)} normal，"
            f"不足 {n_per_class}×2，回退 Neo4j 全库抽样。"
        )
        return sample_test_urls(n_per_class, seed, active_only=active_only)
    rng.shuffle(picked)
    return picked


def _normalize_run(timing_ms: dict, e2e_http: float) -> dict[str, float]:
    """合并服务端分段与 HTTP 端到端（FastAPI/JSON 开销计入 other）。"""
    out = {
        "enrich": float(timing_ms.get("enrich", 0)),
        "merge": float(timing_ms.get("merge", 0)),
        "embed": float(timing_ms.get("embed", 0)),
        "cypher": float(timing_ms.get("cypher", 0)),
        "gnn": float(timing_ms.get("gnn", 0)),
    }
    server_e2e = float(timing_ms.get("e2e", sum(out.values()) + float(timing_ms.get("other", 0))))
    server_other = float(timing_ms.get("other", 0))
    http_overhead = max(0.0, e2e_http - server_e2e)
    out["other"] = server_other + http_overhead
    out["e2e"] = e2e_http
    return out


def summarize(runs: list[dict[str, float]]) -> list[StageStats]:
    e2e_mean = statistics.mean(r["e2e"] for r in runs)
    stats: list[StageStats] = []
    for key, label in STAGE_ROWS:
        vals = [r[key] for r in runs]
        mean_v = statistics.mean(vals)
        std_v = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        share = 100.0 if key == "e2e" else (mean_v / e2e_mean * 100.0 if e2e_mean > 0 else 0.0)
        stats.append(StageStats(label, mean_v, std_v, share))
    return stats


def print_table(stats: list[StageStats], n: int) -> None:
    headers = ["环节", "平均耗时 (ms)", "标准差 (ms)", "占比"]
    rows = [[s.name, f"{s.mean_ms:.1f}", f"{s.std_ms:.1f}", f"{s.share_pct:.1f}%"] for s in stats]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print("\n" + "=" * 78)
    print("表 6.6 在线检测各环节平均耗时（毫秒）— POST /predict 端到端")
    print("=" * 78)
    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in rows:
        print(fmt(row))
    print("=" * 78)
    print(f"有效样本 n={n}；端到端 = TestClient POST /predict 墙钟时间（含 JSON 序列化）。")


def main() -> None:
    p = argparse.ArgumentParser(description="POST /predict 端到端耗时基准（表 6.6）")
    p.add_argument("--per-class", type=int, default=250, help="fraud/normal 各抽样条数（合计 500）")
    p.add_argument("--limit", type=int, default=0, help="最多请求条数（0=per-class×2）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument(
        "--from-test-split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="从 stratified 测试子集抽样（约 101 条）；默认从全库 fraud/normal 各 per-class 条",
    )
    p.add_argument(
        "--active-only",
        action="store_true",
        help="仅抽样 Neo4j 中已有 title/content_hash 的活跃域名",
    )
    p.add_argument("--output", type=Path, default=Path("artifacts/predict_latency_table66.json"))
    args = p.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    n_total = args.limit if args.limit > 0 else args.per_class * 2
    if args.from_test_split:
        urls = sample_test_urls_from_split(args.per_class, args.seed, active_only=args.active_only)
    else:
        urls = sample_test_urls(args.per_class, args.seed, active_only=args.active_only)
    urls = urls[:n_total]

    print(
        f"基准：单线程顺序 POST /predict，n={len(urls)}，"
        f"{'测试集' if args.from_test_split else '全库'}抽样，seed={args.seed}\n"
    )

    client = TestClient(app)
    # 预热模型与嵌入
    embed_texts(["warmup"])
    get_cached_model()

    for i in range(min(args.warmup, len(urls))):
        client.post(
            "/predict",
            json={"url": urls[i]},
            headers={"X-Profile-Latency": "1"},
        )
        print(f"  warmup {i + 1}/{args.warmup}")

    runs: list[dict[str, float]] = []
    failed = 0
    t_batch = time.perf_counter()
    for i, url in enumerate(urls, start=1):
        try:
            t0 = time.perf_counter()
            resp = client.post(
                "/predict",
                json={"url": url},
                headers={"X-Profile-Latency": "1"},
            )
            e2e_http = (time.perf_counter() - t0) * 1000.0
            if resp.status_code != 200:
                failed += 1
                print(f"  [{i}/{len(urls)}] FAIL {resp.status_code} {url[:60]}")
                continue
            body = resp.json()
            timing_ms = body.get("timing_ms") or {}
            runs.append(_normalize_run(timing_ms, e2e_http))
            if i % max(1, len(urls) // 10) == 0 or i == len(urls):
                print(f"  [{i}/{len(urls)}] e2e={e2e_http:.0f} ms  enrich={timing_ms.get('enrich', 0):.0f} ms")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [{i}/{len(urls)}] ERR {url[:50]}: {e}")

    if not runs:
        raise SystemExit("无成功请求，请检查 Neo4j、模型权重与网络。")

    stats = summarize(runs)
    print_table(stats, len(runs))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "n_requested": len(urls),
                "n_ok": len(runs),
                "n_failed": failed,
                "elapsed_sec": time.perf_counter() - t_batch,
                "stats": [
                    {
                        "stage": s.name,
                        "mean_ms": s.mean_ms,
                        "std_ms": s.std_ms,
                        "share_pct": s.share_pct,
                    }
                    for s in stats
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"已保存: {args.output.resolve()}")


if __name__ == "__main__":
    main()
