import json
from pathlib import Path
from app.core.agent.embeddings_factory import get_embeddings
from langchain_redis import RedisVectorStore
from app.core.config import settings

def seed_objections():
    PROJECT_ROOT = Path(__file__).parent.parent
    JSON_PATH = PROJECT_ROOT / "data" / "objection_examples.json"
    
    if not JSON_PATH.exists():
        print(f"JSON not found at {JSON_PATH}. Run the extraction script first.")
        return

    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    
    texts = [item["query"] for item in data]
    metadatas = [{"answer": item["answer"]} for item in data]
    
    # Note: Use the same index name or a specific one for objections
    embeddings = get_embeddings() 

    try:
        vector_store = RedisVectorStore(
            redis_url=settings.REDIS_URL,
            index_name="cato_objections_index",
            embedding=embeddings
        )
        vector_store.drop_index()
        print("Dropped old index for a clean slate.")
    except Exception:
        pass

    vector_store = RedisVectorStore.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        redis_url=settings.REDIS_URL,
        index_name="cato_objections_index"
    )
    
    print(f"Successfully indexed {len(texts)} objections to Redis.")

if __name__ == "__main__":
    seed_objections()