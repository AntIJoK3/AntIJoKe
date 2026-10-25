"""端到端：可选清空库 → schema → Parquet 导入与嵌入 → 图投影与 GDS → 训练 GraphSAGE。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from crypo_side.config import get_settings

_root = Path(__file__).resolve().parents[1]


def run_py(args: list[str]) -> None:
    cmd = [sys.executable, *args]
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(_root))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        type=Path,
        default=get_settings().deepurl_parquet_dir,
        help="Parquet 目录",
    )
    p.add_argument("--wipe", action="store_true", help="清空图数据库（危险，仅开发环境）")
    p.add_argument("--skip-ingest", action="store_true", help="跳过 Parquet 导入（图已存在时）")
    p.add_argument("--skip-gds", action="store_true", help="跳过 homog/similar/GDS 步骤")
    p.add_argument("--skip-train", action="store_true", help="跳过模型训练")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument(
        "--gds-projection",
        choices=["minimal", "full"],
        default="minimal",
        help="传给 neo4j_apply graph_features（默认 minimal 适配 DeepURL）",
    )
    args = p.parse_args()

    if args.wipe:
        run_py([str(_root / "scripts" / "wipe_graph.py"), "--yes"])

    run_py([str(_root / "scripts" / "neo4j_apply.py"), "schema"])
    if not args.skip_ingest:
        run_py(
            [
                str(_root / "scripts" / "ingest_parquet_dataset.py"),
                "--data-dir",
                str(args.data_dir),
            ]
        )
    if not args.skip_gds:
        for step in ("homog", "similar"):
            run_py([str(_root / "scripts" / "neo4j_apply.py"), step])
        run_py(
            [
                str(_root / "scripts" / "neo4j_apply.py"),
                "graph_features",
                "--gds-projection",
                args.gds_projection,
            ]
        )
    if not args.skip_train:
        run_py(["-m", "crypo_side.ml.train", "--epochs", str(args.epochs)])

    print("流水线结束。启动 API: uvicorn crypo_side.api.main:app --reload")


if __name__ == "__main__":
    main()
