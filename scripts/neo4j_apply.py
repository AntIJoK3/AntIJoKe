"""在 Neo4j 上执行 resources/cypher 下的步骤脚本。"""
from __future__ import annotations

import argparse

from crypo_side.db import make_driver
from crypo_side.graph.cypher_runner import resources_dir, run_cypher_file
from crypo_side.graph.feature_fallback import try_run_gds_then_concat

FILE_STEPS = {
    "schema": "01_schema.cypher",
    "homog": "02_project_homogeneous.cypher",
    "similar": "05_content_similar.cypher",
    "feat_concat": "06_feat_vector_concat.cypher",
}

STEP_CHOICES = list(FILE_STEPS.keys()) + ["graph_features", "all", "gds_project", "gds_features"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "step",
        choices=STEP_CHOICES,
        help="graph_features=GDS 或 Python 回退 + feat_vector 拼接；all=schema,homog,similar,graph_features",
    )
    p.add_argument(
        "--gds-projection",
        choices=["minimal", "full"],
        default="minimal",
        help="仅对 graph_features / all 有效：minimal=仅 HAS_IP/SHARED_IP/CONTENT_SIMILAR（DeepURL 默认）；full=含 WHOIS/NS/证书等同构边（图中须已存在对应关系类型）",
    )
    args = p.parse_args()
    base = resources_dir()
    driver = make_driver()
    with driver.session() as session:
        if args.step == "all":
            order = ["schema", "homog", "similar", "graph_features"]
        elif args.step in ("gds_project", "gds_features"):
            print("提示: 请使用 graph_features（自动 GDS 或回退并完成 feat_vector 拼接）")
            order = ["graph_features"]
        else:
            order = [args.step]
        for s in order:
            if s == "graph_features":
                print(
                    "执行: GDS 或 NetworkX 回退 + 06_feat_vector_concat.cypher",
                    f"(GDS 投影: {args.gds_projection})",
                )
                try_run_gds_then_concat(session, base, gds_projection=args.gds_projection)
                continue
            fname = FILE_STEPS.get(s)
            if not fname:
                continue
            path = base / fname
            if not path.exists():
                raise SystemExit(f"缺少文件: {path}")
            print("执行:", path.name)
            run_cypher_file(session, path)
    print("完成")


if __name__ == "__main__":
    main()
