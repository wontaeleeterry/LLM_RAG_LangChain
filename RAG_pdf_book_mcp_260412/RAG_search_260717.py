from pathlib import Path
from embed_store import VectorStore


class RAGEngine:
    def __init__(self):
        self.store = VectorStore()
        self.base_dir = Path(__file__).resolve().parent
        self.is_loaded = False

    def load_db(self):
        if not self.is_loaded:
            db_dir = self.base_dir / "vectordb"

            self.store.load(
                str(db_dir / "faiss.index"),
                str(db_dir / "metadata.pkl")
            )
            self.is_loaded = True

    def ask(self, query: str, top_k: int = 10):
        self.load_db()
        docs = self.store.search(query, top_k=top_k)

        context = "\n\n".join(
            [f"[Page {d.get('page', '?')}] {d.get('text', '')}" for d in docs]
        )
        return context