"""DeepURLBench 风格 Parquet 加载、标签映射与分块迭代。"""
from __future__ import annotations

import ipaddress
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pyarrow.parquet as pq
import tldextract

GRAPH_LABEL_FRAUD = "fraud"
GRAPH_LABEL_NORMAL = "normal"

_EXPECTED_COLS = frozenset({"url", "label", "ip_address"})

# 与 collection/enrich.py 一致：加载随包内置的 PSL 快照，避免联网与缓存文件锁竞争。
_tld = tldextract.TLDExtract(
    suffix_list_urls=(),
    cache_dir=None,
    fallback_to_snapshot=True,
)


def normalize_ip(addr: str) -> str | None:
    s = (addr or "").strip()
    if not s:
        return None
    parts = s.split(".")
    if len(parts) == 4 and all(p.isdigit() or (p.startswith("0") and p[1:].isdigit()) for p in parts):
        try:
            return ".".join(str(int(p)) for p in parts)
        except ValueError:
            return None
    try:
        return str(ipaddress.ip_address(s))
    except ValueError:
        return None


def domain_from_url(url: str) -> str:
    p = urlparse(url.strip() if url else "")
    host = (p.hostname or "").lower()
    ext = _tld(host)
    return ext.registered_domain or host


def map_dataset_label(raw: str) -> str | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip().lower()
    if s in ("phishing", "malware", "defacement", "spam"):
        return GRAPH_LABEL_FRAUD
    if s in ("benign", "normal"):
        return GRAPH_LABEL_NORMAL
    return None


def list_parquet_paths(data_dirs: list[Path], *, recursive: bool = False) -> list[Path]:
    """列出多个目录下的 parquet 文件路径（去重、排序）。"""
    seen: set[Path] = set()
    out: list[Path] = []
    for d in data_dirs:
        if not d.is_dir():
            continue
        it = d.rglob("*.parquet") if recursive else d.glob("*.parquet")
        for p in sorted(it):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(p)
    return sorted(out)


def read_parquet_dataset(data_dir: Path) -> pd.DataFrame:
    """读取单个目录下所有 *.parquet 并合并（小数据用；大数据请用 iter_parquet_batches）。"""
    paths = list_parquet_paths([data_dir], recursive=False)
    if not paths:
        raise FileNotFoundError(f"目录中无 parquet 文件: {data_dir}")
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    missing = _EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Parquet 缺少列: {missing}；当前列: {list(df.columns)}")
    return df


def iter_parquet_batches(paths: list[Path], batch_size: int = 8192):
    """按 Arrow batch 流式产出 DataFrame 块，避免一次性读入千万行。"""
    if not paths:
        raise FileNotFoundError("未找到任何 parquet 路径")
    for path in paths:
        pf = pq.ParquetFile(path)
        missing = _EXPECTED_COLS - set(pf.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} 缺少列 {missing}；列为 {pf.schema_arrow.names}")
        for batch in pf.iter_batches(batch_size=batch_size):
            yield batch.to_pandas()


def rows_for_neo4j(df: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    for _, row in df.iterrows():
        raw_label = row.get("label")
        gl = map_dataset_label(raw_label)
        if gl is None:
            continue
        url = str(row["url"]).strip()
        if not url:
            continue
        # 与 collection/enrich.py 的 _normalize_url 保持一致：图中 Website.url 以 scheme 开头，
        # 否则裸域名 "x.com" 在在线推理时匹配不到图中的 "https://x.com"。
        if not url.lower().startswith(("http://", "https://")):
            url = "http://" + url
        ips_raw = row.get("ip_address")
        if ips_raw is None or (isinstance(ips_raw, float) and pd.isna(ips_raw)):
            ips_list: list = []
        elif hasattr(ips_raw, "tolist"):
            ips_list = list(ips_raw.tolist())
        elif isinstance(ips_raw, (list, tuple)):
            ips_list = list(ips_raw)
        else:
            ips_list = [ips_raw]
        ips: list[str] = []
        for a in ips_list:
            n = normalize_ip(str(a))
            if n:
                ips.append(n)
        fs = row.get("first_seen")
        fs_iso = None
        if fs is not None and not (isinstance(fs, float) and pd.isna(fs)):
            try:
                fs_iso = pd.Timestamp(fs).to_pydatetime().isoformat()
            except Exception:  # noqa: BLE001
                fs_iso = str(fs)
        ttl = row.get("TTL")
        try:
            ttl_i = int(ttl) if ttl is not None and not (isinstance(ttl, float) and pd.isna(ttl)) else None
        except (TypeError, ValueError):
            ttl_i = None
        out.append(
            {
                "url": url,
                "domain": domain_from_url(url),
                "label": gl,
                "source_label": str(raw_label),
                "first_seen": fs_iso,
                "ttl": ttl_i,
                "ips": ips,
            }
        )
    return out
