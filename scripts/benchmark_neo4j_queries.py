"""表 6.7：Neo4j 典型图查询效率（全量 vs 扩展图）。"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from crypo_side.db import make_driver
from crypo_side.graph.rel_types import website_homog_edge_types


@dataclass
class QueryBench:
    key: str
    label: str
    full_ms: float
    ext_ms: float

    @property
    def growth_pct(self) -> float:
        if self.full_ms <= 0:
            return 0.0
        return (self.ext_ms - self.full_ms) / self.full_ms * 100.0


def _ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def _rel_types() -> str:
    return website_homog_edge_types()


def _is_ext_node(alias: str = "w") -> str:
    return f"{alias}.url STARTS WITH 'benchscale:'"


def _is_full_node(alias: str = "w") -> str:
    return f"NOT ({alias}.url STARTS WITH 'benchscale:')"


def _graph_stats(session, *, extended: bool) -> dict[str, int]:
    rel = _rel_types()
    if extended:
        node_q = "MATCH (w:Website) RETURN count(w) AS n"
        edge_q = f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n"
    else:
        node_q = f"MATCH (w:Website) WHERE {_is_full_node('w')} RETURN count(w) AS n"
        edge_q = (
            f"MATCH (a:Website)-[r:{rel}]-(b:Website) "
            f"WHERE {_is_full_node('a')} AND {_is_full_node('b')} RETURN count(r) AS n"
        )
    nodes = session.run(node_q).single()["n"]
    edges = session.run(edge_q).single()["n"]
    return {"nodes": int(nodes), "edges": int(edges)}


def _pick_seed_urls(session, n: int, *, extended: bool) -> list[str]:
    rel = _rel_types()
    filt = "" if extended else f"WHERE {_is_full_node('w')}"
    q = f"""
    MATCH (w:Website)
    {filt}
    OPTIONAL MATCH (w)-[:{rel}]-(n)
    WITH w, count(n) AS deg
    ORDER BY deg DESC, rand()
    LIMIT $n
    RETURN w.url AS url
    """
    rows = session.run(q, n=n).data()
    return [r["url"] for r in rows]


def _largest_community(session, *, extended: bool) -> int:
    if extended:
        q = """
        MATCH (w:Website)
        WHERE w.louvain_community IS NOT NULL
        WITH w.louvain_community AS comm, count(*) AS n
        ORDER BY n DESC
        LIMIT 1
        RETURN comm AS comm
        """
    else:
        q = f"""
        MATCH (w:Website)
        WHERE {_is_full_node('w')} AND w.louvain_community IS NOT NULL
        WITH w.louvain_community AS comm, count(*) AS n
        ORDER BY n DESC
        LIMIT 1
        RETURN comm AS comm
        """
    row = session.run(q).single()
    if row is None:
        raise RuntimeError("图中无 louvain_community，请先运行 GDS 特征写入。")
    return int(row["comm"])


def _bench_subgraph_1hop(session, url: str, *, extended: bool) -> None:
    rel = _rel_types()
    if extended:
        q = f"""
        MATCH (w:Website {{url: $url}})
        MATCH (w)-[r:{rel}]-(m)
        WITH type(r) AS rt, w, m
        LIMIT 120
        RETURN elementId(w) AS wid, labels(w) AS lw, properties(w) AS pw,
               elementId(m) AS mid, labels(m) AS lm, properties(m) AS pm,
               rt
        """
    else:
        q = f"""
        MATCH (w:Website {{url: $url}})
        WHERE {_is_full_node('w')}
        MATCH (w)-[r:{rel}]-(m)
        WHERE {_is_full_node('m')}
        WITH type(r) AS rt, w, m
        LIMIT 120
        RETURN elementId(w) AS wid, labels(w) AS lw, properties(w) AS pw,
               elementId(m) AS mid, labels(m) AS lm, properties(m) AS pm,
               rt
        """
    session.run(q, url=url).consume()


def _bench_community_large(session, comm: int, *, extended: bool) -> None:
    rel = _rel_types()
    if extended:
        q = f"""
        MATCH (w:Website {{louvain_community: $comm}})
        WHERE w.label IN ['fraud', 'normal']
        WITH collect(w) AS members
        WITH members,
             size(members) AS node_cnt,
             size([m IN members WHERE m.label = 'fraud']) AS fraud_cnt
        UNWIND members AS m
        OPTIONAL MATCH (m)-[:{rel}]-(x:Website)
        WHERE x.louvain_community <> $comm
        WITH node_cnt, fraud_cnt, count(DISTINCT x) AS cross_comm_edges
        RETURN node_cnt, fraud_cnt, cross_comm_edges,
               toFloat(fraud_cnt) / node_cnt AS cfr
        """
    else:
        q = f"""
        MATCH (w:Website {{louvain_community: $comm}})
        WHERE {_is_full_node('w')} AND w.label IN ['fraud', 'normal']
        WITH collect(w) AS members
        WITH members,
             size(members) AS node_cnt,
             size([m IN members WHERE m.label = 'fraud']) AS fraud_cnt
        UNWIND members AS m
        OPTIONAL MATCH (m)-[:{rel}]-(x:Website)
        WHERE {_is_full_node('x')} AND x.louvain_community <> $comm
        WITH node_cnt, fraud_cnt, count(DISTINCT x) AS cross_comm_edges
        RETURN node_cnt, fraud_cnt, cross_comm_edges,
               toFloat(fraud_cnt) / node_cnt AS cfr
        """
    session.run(q, comm=comm).consume()


def _bench_two_hop(session, url: str, *, extended: bool) -> None:
    rel = _rel_types()
    if extended:
        q = f"""
        MATCH (w:Website {{url: $url}})
        MATCH (w)-[:{rel}*1..2]-(n:Website)
        WHERE n <> w
        RETURN count(DISTINCT n) AS two_hop_cnt
        """
    else:
        q = f"""
        MATCH (w:Website {{url: $url}})
        WHERE {_is_full_node('w')}
        MATCH (w)-[:{rel}*1..2]-(n:Website)
        WHERE n <> w AND {_is_full_node('n')}
        RETURN count(DISTINCT n) AS two_hop_cnt
        """
    session.run(q, url=url).consume()


def _run_many(fn, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(_ms(t0, time.perf_counter()))
    return out


def materialize_scaled_graph(session, *, cleanup: bool = False) -> None:
    rel_types = _rel_types().split("|")
    if cleanup:
        for rt in rel_types:
            session.run(
                f"MATCH ()-[r:{rt}]->() WHERE startNode(r).url STARTS WITH 'benchscale:' DELETE r"
            ).consume()
        session.run(
            "MATCH (w:Website) WHERE w.url STARTS WITH 'benchscale:' DETACH DELETE w"
        ).consume()
        return

    exists = session.run(
        "MATCH (w:Website) WHERE w.url STARTS WITH 'benchscale:' RETURN count(w) AS n LIMIT 1"
    ).single()["n"]
    if exists:
        return

    session.run(
        f"""
        MATCH (w:Website)
        WHERE {_is_full_node('w')}
        WITH w
        CREATE (w2:Website:BenchScaleExt)
        SET w2.url = 'benchscale:' + w.url,
            w2.bench_src_url = w.url,
            w2.domain = w.domain,
            w2.title = w.title,
            w2.content_hash = w.content_hash,
            w2.label = w.label,
            w2.louvain_community = w.louvain_community,
            w2.pagerank = w.pagerank,
            w2.betweenness = w.betweenness,
            w2.gnn_fraud_prob = w.gnn_fraud_prob
        """
    ).consume()
    for rt in rel_types:
        session.run(
            f"""
            MATCH (a:Website)-[:{rt}]-(b:Website)
            WHERE {_is_full_node('a')} AND {_is_full_node('b')}
            MATCH (a2:Website {{bench_src_url: a.url}})
            MATCH (b2:Website {{bench_src_url: b.url}})
            MERGE (a2)-[:{rt}]-(b2)
            """
        ).consume()


def benchmark_suite(repeats: int, warmup: int, samples: int) -> tuple[list[QueryBench], dict]:
    driver = make_driver()
    with driver.session() as session:
        materialize_scaled_graph(session)
        full_stats = _graph_stats(session, extended=False)
        ext_stats = _graph_stats(session, extended=True)

        seed_full = _pick_seed_urls(session, samples, extended=False)
        seed_ext = _pick_seed_urls(session, samples, extended=True)
        comm_full = _largest_community(session, extended=False)
        comm_ext = _largest_community(session, extended=True)

        results: list[QueryBench] = []

        def avg(fn) -> float:
            return statistics.mean(_run_many(fn, repeats=repeats, warmup=warmup))

        full_1hop = avg(lambda: [_bench_subgraph_1hop(session, u, extended=False) for u in seed_full])
        ext_1hop = avg(lambda: [_bench_subgraph_1hop(session, u, extended=True) for u in seed_ext])
        results.append(QueryBench("subgraph_1hop", "子图导出（1-hop）", full_1hop, ext_1hop))

        full_comm = avg(lambda: _bench_community_large(session, comm_full, extended=False))
        ext_comm = avg(lambda: _bench_community_large(session, comm_ext, extended=True))
        results.append(QueryBench("community_large", "社区查询（大型社区）", full_comm, ext_comm))

        full_2hop = avg(lambda: [_bench_two_hop(session, u, extended=False) for u in seed_full])
        ext_2hop = avg(lambda: [_bench_two_hop(session, u, extended=True) for u in seed_ext])
        results.append(QueryBench("two_hop", "多跳关联（2-hop）", full_2hop, ext_2hop))

        meta = {
            "full_graph": full_stats,
            "extended_graph": ext_stats,
            "largest_community_full": comm_full,
            "largest_community_ext": comm_ext,
            "samples_per_query": samples,
            "repeats": repeats,
            "warmup": warmup,
        }
        return results, meta


def print_table(rows: list[QueryBench]) -> None:
    headers = ["查询类型", "全量数据平均耗时 (ms)", "扩展数据平均耗时 (ms)", "增长率"]
    body = [
        [r.label, f"{r.full_ms:.1f}", f"{r.ext_ms:.1f}", f"{r.growth_pct:.1f}%"]
        for r in rows
    ]
    widths = [max(len(h), *(len(row[i]) for row in body)) for i, h in enumerate(headers)]

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print("\n" + "=" * 72)
    print("表 6.7 图数据库典型查询效率")
    print("=" * 72)
    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in body:
        print(fmt(row))
    print("=" * 72)


def main() -> None:
    p = argparse.ArgumentParser(description="Neo4j 典型查询效率基准（表 6.7）")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--cleanup-ext", action="store_true")
    p.add_argument("--output", type=Path, default=Path("artifacts/neo4j_query_table67.json"))
    args = p.parse_args()

    driver = make_driver()
    if args.cleanup_ext:
        with driver.session() as session:
            materialize_scaled_graph(session, cleanup=True)
        print("已清理扩展图节点（benchscale:*）。")
        return

    rows, meta = benchmark_suite(args.repeats, args.warmup, args.samples)
    print_table(rows)
    print(
        f"\n全量图：{meta['full_graph']['nodes']} 节点 / {meta['full_graph']['edges']} 同构边；"
        f"扩展图：{meta['extended_graph']['nodes']} 节点 / {meta['extended_graph']['edges']} 同构边。"
    )
    print(
        f"最大 Louvain 社区：全量 comm={meta['largest_community_full']}，"
        f"扩展 comm={meta['largest_community_ext']}；"
        f"repeats={meta['repeats']}，seeds={meta['samples_per_query']}。"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "meta": meta,
                "rows": [
                    {
                        "query": r.label,
                        "key": r.key,
                        "full_mean_ms": r.full_ms,
                        "ext_mean_ms": r.ext_ms,
                        "growth_pct": r.growth_pct,
                    }
                    for r in rows
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
