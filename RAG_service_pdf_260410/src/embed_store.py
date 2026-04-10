import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []

    def add_documents(self, docs: list):
        texts = [doc["text"] for doc in docs]
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings).astype("float32"))
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
        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"),
            top_k,
        )

        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results