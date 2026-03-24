from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def create_collection(collection_name: str) -> bool:
    client = QdrantClient(url="http://localhost:6333")
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=4, distance=Distance.DOT)
        )
        return True
    except Exception as e:
        return False