"""一次性校验本项目用到的第三方包是否已安装（本地/CI 可运行）。"""
from __future__ import annotations

import importlib
import sys


def _try(name: str, import_path: str | None = None) -> None:
    mod = import_path or name
    try:
        importlib.import_module(mod)
    except ImportError as e:
        print(f"FAIL  {name}  (import {mod!r}): {e}", file=sys.stderr)
        raise


def main() -> None:
    # 包名（pip）与 import 模块名不一致的单独写
    _try("neo4j")
    _try("httpx")
    _try("bs4", "bs4")
    _try("lxml", "lxml")
    _try("dotenv", "dotenv")
    _try("pydantic_settings", "pydantic_settings")
    _try("pandas", "pandas")
    _try("sklearn", "sklearn")
    _try("numpy", "numpy")
    _try("tldextract", "tldextract")
    _try("pyarrow", "pyarrow")
    _try("tqdm", "tqdm")
    _try("torch", "torch")
    _try("torch_geometric", "torch_geometric")
    _try("sentence_transformers", "sentence_transformers")
    _try("networkx", "networkx")
    _try("dnspython", "dns")  # PyPI: dnspython
    _try("python-whois", "whois")
    _try("fastapi", "fastapi")
    _try("uvicorn", "uvicorn")

    import crypo_side.collection.enrich  # noqa: F401 — 触发 dns/httpx/bs4 等链式导入
    import crypo_side.ml.graphsage  # noqa: F401
    import crypo_side.ml.embeddings  # noqa: F401

    print("OK: all listed deps and critical crypo_side submodules import successfully.")


if __name__ == "__main__":
    main()
