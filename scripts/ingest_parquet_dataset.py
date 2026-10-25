"""从 Parquet 导入 Neo4j 并生成文本嵌入（支持多目录、分块流式、可选 HTTP 丰富）。"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from crypo_side.collection.enrich import enrich_url
from crypo_side.config import get_settings
from crypo_side.dataset.deepurl import iter_parquet_batches, list_parquet_paths, rows_for_neo4j
from crypo_side.db import make_driver
from crypo_side.graph.bulk_ingest import merge_website_batch, set_text_embeddings_batch
from crypo_side.graph.importer import merge_website_graph
from crypo_side.ml.embeddings import embed_texts


def chunked(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def main() -> None:
    p = argparse.ArgumentParser(
        description="从 Parquet 批量写入 Website/IP 并生成 text_embedding；大数据用 --chunk-rows 流式读取。"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        dest="data_dirs",
        default=None,
        help="含 parquet 的目录，可重复指定多个；不指定则用配置 deepurl_parquet_dir",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="在每个 --data-dir 下递归查找 *.parquet",
    )
    p.add_argument(
        "--chunk-rows",
        type=int,
        default=8192,
        help="Arrow 每批行数（流式读取，控制内存）",
    )
    p.add_argument("--write-batch", type=int, default=200, help="每批 UNWIND 写入 Neo4j 行数")
    p.add_argument("--embed-batch", type=int, default=64, help="sentence-transformers 批大小")
    p.add_argument(
        "--http-enrich",
        action="store_true",
        help="对每条 URL 再拉首页/DNS/WHOIS 等写入图（慢；配合 --http-enrich-limit 做小样本）",
    )
    p.add_argument(
        "--http-enrich-limit",
        type=int,
        default=0,
        help="仅前 N 条走 HTTP 丰富，之后同批及后续批次仅用 Parquet 写库；0 表示不限制",
    )
    args = p.parse_args()

    dirs = args.data_dirs if args.data_dirs else [get_settings().deepurl_parquet_dir]
    paths = list_parquet_paths(dirs, recursive=args.recursive)
    if not paths:
        raise SystemExit(f"未找到 parquet：{dirs}")

    driver = make_driver()
    total_rows = 0
    http_attempts = 0
    enrich_limit = args.http_enrich_limit if args.http_enrich_limit > 0 else None

    for df_chunk in tqdm(iter_parquet_batches(paths, batch_size=args.chunk_rows), desc="Parquet 块"):
        rows = rows_for_neo4j(df_chunk)
        if not rows:
            continue

        pairs: list[dict] = []
        parquet_only: list[dict] = []

        if args.http_enrich:
            with driver.session() as session:
                for row in tqdm(rows, desc="HTTP 丰富+写库", leave=False):
                    use_http = enrich_limit is None or http_attempts < enrich_limit
                    if not use_http:
                        parquet_only.append(row)
                        continue
                    try:
                        en = enrich_url(row["url"])
                        merge_website_graph(session, en, row["label"])
                        http_attempts += 1
                        text = (en.visible_text or en.title or en.domain or "").strip()
                        if not text:
                            text = f"{row['url']} {row['domain']}"
                        pairs.append({"url": en.url, "text": text[:12000]})
                    except Exception:  # noqa: BLE001
                        merge_website_batch(session, [row])
                        http_attempts += 1
                        pairs.append({"url": row["url"], "text": f"{row['url']} {row['domain']}"})
            if parquet_only:
                with driver.session() as session:
                    for batch in chunked(parquet_only, args.write_batch):
                        merge_website_batch(session, batch)
                pairs.extend(
                    {"url": r["url"], "text": f"{r['url']} {r['domain']}"} for r in parquet_only
                )
        else:
            with driver.session() as session:
                for batch in chunked(rows, args.write_batch):
                    merge_website_batch(session, batch)
            pairs = [{"url": r["url"], "text": f"{r['url']} {r['domain']}"} for r in rows]

        total_rows += len(rows)

        for i in tqdm(range(0, len(pairs), args.embed_batch), desc="文本嵌入", leave=False):
            chunk = pairs[i : i + args.embed_batch]
            vecs = embed_texts([c["text"] for c in chunk])
            out = [{"url": c["url"], "vec": v} for c, v in zip(chunk, vecs, strict=True)]
            with driver.session() as session:
                for b in chunked(out, args.write_batch):
                    set_text_embeddings_batch(session, b)

    print(
        "完成：",
        total_rows,
        "行（有效标签）已写库并嵌入；HTTP 丰富尝试次数:",
        http_attempts if args.http_enrich else 0,
        "。下一步：python scripts/neo4j_apply.py homog similar graph_features 后训练。",
    )


if __name__ == "__main__":
    main()
