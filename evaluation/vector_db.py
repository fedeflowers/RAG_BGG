from abc import ABC
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore



class VectorDB(ABC):

    def __init__(self,url,api_key):
        self.url = url
        self.api_key = api_key

    def scroll(self,collection_name,filter=None):
        pass

    def get_retriever(self,collection_name,embeddings):
        pass



class QdrantVectorDB(VectorDB):

    def __init__(self, url,api_key):
        super().__init__(url,api_key)
        self.client = QdrantClient(url=self.url,api_key=self.api_key)

    def scroll(self,collection_name,filter=None):
        points = self.client.scroll(collection_name=collection_name,scroll_filter=filter)

        chunks = []
        while points:
            for point in points[0]:
                payload = point.payload
                chunks.append({"page_content":payload["page_content"],"metadata":payload["metadata"]})
            if points[1]:
                points = self.client.scroll(collection_name=collection_name,offset=points[1],scroll_filter=filter)
            else:
                break
        return chunks
    
    def get_retriever(self, collection_name,embeddings):
        vector_db = QdrantVectorStore.from_existing_collection(collection_name=collection_name,embedding=embeddings,url=self.url,api_key=self.api_key)
        retriever = vector_db.as_retriever(search_kwargs={"k":5,"fetch_k":10,"lambda_mult":0.9},search_type="mmr")
        return retriever