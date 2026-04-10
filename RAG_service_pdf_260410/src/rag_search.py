import requests
from embed_store import VectorStore


class RAGEngine:
    def __init__(self, model_name: str = "llama3"):
        self.store = VectorStore()
        self.model_name = model_name

    def load_db(self):
        self.store.load(
            "vectordb/faiss.index",
            "vectordb/metadata.pkl",
        )

    def _ask_ollama(self, prompt: str) -> str:
        url = "http://127.0.0.1:11434/api/generate"

        response = requests.post(
            url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )

        response.raise_for_status()
        data = response.json()
        return data.get("response", "응답 생성 실패")

    def ask(self, query: str) -> str:
        docs = self.store.search(query, top_k=3)

        if not docs:
            return "관련 문서를 찾지 못했습니다."

        context = "\n\n---\n\n".join(
            f"[Page {d['page']}]\n{d['text']}"
            for d in docs
        )

        prompt = f"""You are a PDF-based RAG assistant.
        Please answer in English using only the context below as evidence. 
        And add translated anwer in Korean below.

[context]
{context}

[question]
{query}
"""

        return self._ask_ollama(prompt)