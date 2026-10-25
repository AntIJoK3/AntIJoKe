"""清空 Neo4j 中所有节点与关系（仅用于开发/重建流水线）。"""
from __future__ import annotations

import argparse

from crypo_side.db import make_driver


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--yes", action="store_true", help="确认执行删除")
    args = p.parse_args()
    if not args.yes:
        raise SystemExit("请传入 --yes 以确认清空图数据库")
    driver = make_driver()
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    finally:
        driver.close()
    print("已清空")


if __name__ == "__main__":
    main()
