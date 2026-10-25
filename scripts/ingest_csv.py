"""从 CSV 导入种子站点：列 url,label（fraud/normal）。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from crypo_side.collection.enrich import enrich_url
from crypo_side.db import make_driver
from crypo_side.graph.importer import merge_website_graph, set_text_embedding
from crypo_side.ml.embeddings import embed_texts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", type=Path, help="含 url,label 的 CSV")
    args = p.parse_args()
    rows: list[dict] = []
    with args.csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    driver = make_driver()
    for row in tqdm(rows):
        url = row["url"].strip()
        label = row.get("label") or row.get("Label")
        data = enrich_url(url)
        with driver.session() as session:
            merge_website_graph(session, data, label)
        text = data.visible_text or data.domain
        vec = embed_texts([text])[0]
        with driver.session() as session:
            set_text_embedding(session, data.url, vec)

    print("已导入", len(rows), "条；请运行 scripts/neo4j_apply.py homog similar gds_project gds_features")


if __name__ == "__main__":
    main()
