import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384

        # L2 거리 -> Inner Product
        self.index = faiss.IndexFlatIP(self.dimension)  # Cosine Similarity 방식으로 변경 (260414)
        self.metadata = []

    def add_documents(self, docs: list):
        texts = [doc["text"] for doc in docs]

        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype("float32")

        # 코사인 유사도를 위한 L2 정규화
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.metadata.extend(docs)

    def save(self, index_path: str, meta_path: str):
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, index_path: str, meta_path: str):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query: str, top_k: int = 3):
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        # query도 반드시 정규화
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(score)  # cosine similarity score
                results.append(result)

        return results