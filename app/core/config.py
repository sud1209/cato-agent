from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml
import os


@dataclass
class LLMConfig:
    model: str
    temperature: float
    streaming: bool


@dataclass
class EmbeddingsConfig:
    model: str


@dataclass
class RAGConfig:
    retrieval_k: int
    rerank_top_k: int
    reranker: str  # "cohere" | "local"


@dataclass
class MemoryConfig:
    working_window: int
    summary_threshold: int
    profile_ttl_days: int


@dataclass
class RedisConfig:
    url: str


@dataclass
class LangfuseConfig:
    enabled: bool
    public_key: str
    secret_key: str


@dataclass
class Settings:
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    rag: RAGConfig
    memory: MemoryConfig
    redis: RedisConfig
    langfuse: LangfuseConfig


def _load_settings() -> Settings:
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    if not config_path.exists():
        config_path = Path.cwd() / "config.yaml"
    if not config_path.exists():
        primary = Path(__file__).parent.parent.parent / "config.yaml"
        raise FileNotFoundError(
            f"config.yaml not found. Searched:\n"
            f"  1. {primary}\n"
            f"  2. {Path.cwd() / 'config.yaml'}\n"
            f"Create config.yaml at the cato-agent project root."
        )
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    def _env(val: str) -> str:
        if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
            return os.environ.get(val[2:-1], "")
        return val

    lf = raw.get("langfuse", {})
    return Settings(
        llm=LLMConfig(**raw["llm"]),
        embeddings=EmbeddingsConfig(**raw["embeddings"]),
        rag=RAGConfig(**raw["rag"]),
        memory=MemoryConfig(**raw["memory"]),
        redis=RedisConfig(**raw["redis"]),
        langfuse=LangfuseConfig(
            enabled=lf.get("enabled", False),
            public_key=_env(lf.get("public_key", "")),
            secret_key=_env(lf.get("secret_key", "")),
        ),
    )


settings = _load_settings()
