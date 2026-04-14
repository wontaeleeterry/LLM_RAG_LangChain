from pdf_parser import parse_pdf
from chunker import chunk_text
from embed_store import VectorStore
from RAG_search import RAGEngine
# import textwrap


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


if __name__ == "__main__":
    pdf_path = "data/sample.pdf"

    build_vectordb(pdf_path)