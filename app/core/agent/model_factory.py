from langchain.chat_models import init_chat_model
from app.core.config import settings

def get_model(temperature: float = 0):
    """
    Returns a unified model interface. 
    Switches between OpenAI, Anthropic, or Ollama (Llama) 
    based on the MODEL_PROVIDER in your .env.
    """
    
    if settings.MODEL_PROVIDER == "openai":
        model_name = settings.OPENAI_MODEL
    elif settings.MODEL_PROVIDER == "anthropic":
        model_name = settings.ANTHROPIC_MODEL
    else:
        model_name = settings.OLLAMA_MODEL

    return init_chat_model(
        model=model_name,
        model_provider=settings.MODEL_PROVIDER,
        temperature=temperature
    )