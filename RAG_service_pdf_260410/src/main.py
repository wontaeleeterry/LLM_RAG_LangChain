from pdf_parser import parse_pdf
from chunker import chunk_text
from embed_store import VectorStore
from rag_search import RAGEngine
import textwrap


def build_vectordb(pdf_path: str):
    pages = parse_pdf(pdf_path)

    docs = []
    for page in pages:
        chunks = chunk_text(page["text"])
        for chunk in chunks:
            docs.append({
                "page": page["page"],
                "text": chunk
            })

    store = VectorStore()
    store.add_documents(docs)
    store.save(
        "vectordb/faiss.index",
        "vectordb/metadata.pkl"
    )

    print("벡터 DB 저장 완료")


def run_rag():
    rag = RAGEngine()
    rag.load_db()

    while True:
        query = input("질문 입력 (종료: exit): ")
        if query.lower() == "exit":
            break

        answer = rag.ask(query)
        #print(answer)
        #print(textwrap.fill(answer.replace("\\n", "\n"), width=100))
        print(textwrap.fill(answer, width=100))

if __name__ == "__main__":
    pdf_path = "data/sample.pdf"

    build_vectordb(pdf_path)
    run_rag()
