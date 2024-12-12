import asyncio
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, field
from pymilvus import MilvusClient
from lightrag.utils import logger
from .base import BaseVectorStorage

@dataclass
class MilvusVectorStorage(BaseVectorStorage):
    _max_batch_size: int = 100

    @staticmethod
    def create_collection_if_not_exist(client, collection_name: str, **kwargs):
        if client.has_collection(collection_name):
            return
        # TODO add constants for ID max length to 32
        client.create_collection(
            collection_name, max_length=64, id_type="string", **kwargs
        )
        logger.info(f"Collection {collection_name} created")

    def __post_init__(self):
        self._client = MilvusClient(uri="http://milvus:19530")
        logger.info("Milvus client created")
        self.create_collection_if_not_exist(self._client, self.namespace, dimension=self.embedding_func.embedding_dim)

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> Dict:
        if not data:
            logger.warning("No data to insert")
            return {}

        logger.info(f"Milvus Inserting {len(data)} vectors to {self.namespace}")
        
        list_data = []
        contents = []
        
        for doc_id, doc_data in data.items():
            if "content" not in doc_data:
                logger.warning(f"Skipping document {doc_id}: no content field")
                continue
                
            meta_data = {
                "id": doc_id,
                **{k: v for k, v in doc_data.items() if k in self.meta_fields}
            }
            list_data.append(meta_data)
            contents.append(doc_data["content"])

        if not contents:
            logger.warning("No valid content to process")
            return {}

        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)

        for i, d in enumerate(list_data):
            d["vector"] = embeddings[i].tolist()

        try:
            results = self._client.upsert(
                collection_name=self.namespace,
                data=list_data
            )
            return results
        except Exception as e:
            logger.error(f"Error during upsert: {str(e)}")
            raise

    async def query(self, query: str, top_k: int = 5) -> List[Dict]:
        try:
            logger.info(f"Querying {self.namespace} with query: {query}")
            embedding = await self.embedding_func([query])
            results = self._client.search(
                collection_name=self.namespace,
                data=embedding.tolist(),
                limit=top_k,
                output_fields=list(self.meta_fields),
                search_params={
                    "metric_type": "COSINE",
                    "params": {"nprobe": 10}
                }
            )

            if not results or not results[0]:
                return []
            
            logger.debug(f"Query results: {results[0]}")

            return [{
                "id": hit['id'],
                "distance": hit['distance'],
                **hit['entity']
            } for hit in results[0]]

        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            raise
