import os
from langchain.embeddings import init_embeddings
from app.core.config import settings

def get_embeddings():
    """
    Returns an embedding model based on the MODEL_PROVIDER in .env
    """
    provider_map = {
        "openai": "openai",
        "anthropic": "openai",
        "ollama": "ollama"
    }
    
    provider = provider_map.get(settings.MODEL_PROVIDER, "openai")
    
    model_map = {
        "openai": "text-embedding-3-small",
        "ollama": "llama3.2" 
    }
    
    model_name = model_map.get(provider)

    return init_embeddings(
        model=model_name,
        provider=provider
    )