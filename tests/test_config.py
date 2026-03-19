import pytest
from pathlib import Path

def test_config_loads_yaml(tmp_path, monkeypatch):
    """Settings should load values from a config.yaml file."""
    yaml_content = """
llm:
  model: "openai/gpt-4o"
  temperature: 0.7
  streaming: true
embeddings:
  model: "openai/text-embedding-3-large"
rag:
  retrieval_k: 10
  rerank_top_k: 3
  reranker: "local"
memory:
  working_window: 20
  summary_threshold: 16
  profile_ttl_days: 30
redis:
  url: "redis://localhost:6379"
langfuse:
  enabled: false
  public_key: "test-pub"
  secret_key: "test-sec"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)
    monkeypatch.chdir(tmp_path)

    import importlib
    import app.core.config as config_module
    importlib.reload(config_module)
    s = config_module.settings

    assert s.llm.model == "openai/gpt-4o"
    assert s.llm.temperature == 0.7
    assert s.llm.streaming is True
    assert s.rag.retrieval_k == 10
    assert s.rag.reranker == "local"
    assert s.memory.working_window == 20
    assert s.redis.url == "redis://localhost:6379"
    assert s.langfuse.enabled is False
