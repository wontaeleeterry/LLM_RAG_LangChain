from embed_store import VectorStore


class RAGEngine:
    def __init__(self):
        self.store = VectorStore()

    def load_db(self):
        self.store.load(
            "vectordb/faiss.index",
            "vectordb/metadata.pkl"
        )

    def ask(self, query: str):
        docs = self.store.search(query)

        context = "\n\n".join([
            f"[Page {d['page']}] {d['text']}" for d in docs
        ])

        response = f"""
질문: {query}

검색된 문서 기반 답변:
{context}

최종 요약:
위 문서 내용을 기반으로 질문에 답변했습니다.
"""
        return response