from __future__ import annotations

import functools
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from crypo_side.config import get_settings

_HF_HUB_LIBRARY_DEFAULT_RETRIES = 5
_hf_retries_configured = False


def _apply_hf_hub_max_retries(configured_max_retries: int) -> None:
    """将 huggingface_hub 默认 HTTP 重试从 5 次降为配置值（默认 3）。"""
    global _hf_retries_configured
    if _hf_retries_configured:
        return

    import huggingface_hub.utils._http as hf_http

    original = hf_http._http_backoff_base

    @functools.wraps(original)
    def _patched_backoff(method, url, *, max_retries=_HF_HUB_LIBRARY_DEFAULT_RETRIES, **kwargs):
        effective = (
            configured_max_retries
            if max_retries == _HF_HUB_LIBRARY_DEFAULT_RETRIES
            else max_retries
        )
        yield from original(method, url, max_retries=effective, **kwargs)

    hf_http._http_backoff_base = _patched_backoff
    _hf_retries_configured = True


@lru_cache
def _model(name: str) -> SentenceTransformer:
    import os

    settings = get_settings()
    _apply_hf_hub_max_retries(settings.hf_hub_max_retries)

    local_only = os.environ.get("HF_HUB_OFFLINE", "") == "1"
    return SentenceTransformer(name, local_files_only=local_only)


def embed_texts(texts: list[str]) -> list[list[float]]:
    s = get_settings()
    model = _model(s.embedding_model)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [v.astype("float64").tolist() for v in vectors]
