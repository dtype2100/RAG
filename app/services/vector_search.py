from qdrant_client import QdrantClient
from typing import List, Dict, Any
from qdrant_client.models import Filter, Filter, FieldCondition, MatchValue


class VectorSearch:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client = QdrantClient(url="http://localhost:6333")

    def vector_search(self, query: List[float]) -> List[Dict[str, Any]]:
        """
        Vector Search
        """
        search_results = self.client.query_point(
            collection_name=self.collection_name,
            query=query,
            with_payload=False,
        ).points
        return search_results

    def vector_search_with_filter(self, query: List[float], filter_key: str, filter_value: str) -> List[Dict[str, Any]]:
        """
        Vector Search with Filter
        """
        search_results = self.client.query_point(
            collection_name=self.collection_name,
            query=query,
            query_filter=Filter(
                must=[FieldCondition(key=filter_key, match=MatchValue(value=filter_value))]
            ),
            with_payload=True,
        ).points
        return search_results
    
    def vector_search_with_filter_and_limit(self, query: List[float], filter_key: str, filter_value: str, limit: int) -> List[Dict[str, Any]]:
        """
        Vector Search with Filter and Limit
        """
        search_results = self.client.query_point(
            collection_name=self.collection_name,
            query=query,
            query_filter=Filter(
                must=[FieldCondition(key=filter_key, match=MatchValue(value=filter_value))]
            ),
            with_payload=True,
            limit=limit,
        ).points
        return search_results