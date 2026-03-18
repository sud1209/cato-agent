from langchain_core.tools import tool
from app.db.redis_client import redis_manager
from langchain_openai import OpenAIEmbeddings

@tool
def check_qualification_rules(query: str) -> str:
    """
    Search the HEI knowledge base for qualification rules, Nada's criteria, 
    FICO requirements, and lien policies. Use this for 'Qualifier' or 'Objection' logic.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = redis_manager.get_vector_store(embeddings)
    
    docs = vectorstore.similarity_search(query, k=3)
    
    # Return formatted context for the agent to reason over
    return "\n\n".join([doc.page_content for doc in docs])