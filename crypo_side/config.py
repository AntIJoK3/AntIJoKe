from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

_root = Path(__file__).resolve().parents[1]
load_dotenv(_root / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    deepurl_parquet_dir: Path = _root / "urls_with_dns_live_mixed" / "parts"
    vector_index_name: str = "content_vec_idx"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    gnn_checkpoint: Path = _root / "artifacts" / "graphsage.pt"
    random_state: int = 42
    hf_hub_max_retries: int = 3
    fraud_alert_threshold: float = 0.7

    # Website—Website 用于 GNN 的边类型（竖线分隔）；扩展前请确保图中已有对应关系
    graph_website_homog_edges: str = "SHARED_IP|CONTENT_SIMILAR"


@lru_cache
def get_settings() -> Settings:
    return Settings()
