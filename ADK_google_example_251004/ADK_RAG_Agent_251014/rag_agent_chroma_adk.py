import os
import glob
import PyPDF2
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

# ==============================
# RAG Agent with ChromaDB (Persistent) + Google GenAI (Gemini)
# ==============================

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def build_vector_store(doc_folder="./doc", db_path="./chroma_db"):
    print(f"📚 Building ChromaDB persistent store from: {doc_folder}")

    # 최신 방식으로 PersistentClient 사용
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="pdf_docs")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    pdf_files = glob.glob(os.path.join(doc_folder, "*.pdf"))
    if not pdf_files:
        print("⚠️ No PDF files found in ./doc folder.")
        return

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        text = extract_text_from_pdf(pdf_path)
        chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
        embeddings = embedder.encode(chunks, convert_to_numpy=True).tolist()

        ids = [f"{os.path.basename(pdf_path)}_{i}" for i in range(len(chunks))]
        collection.add(ids=ids, documents=chunks, embeddings=embeddings)

    print("✅ Vector store built successfully and saved persistently.")


def chat_with_agent(query, db_path="./chroma_db"):
    print(f"💬 Query: {query}")

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="pdf_docs")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = embedder.encode(query).tolist()

    results = collection.query(query_embeddings=[query_emb], n_results=3)
    docs = results.get("documents", [[]])[0]
    if not docs:
        print("⚠️ No relevant documents found.")
        return

    context = "\n".join(docs)

    try:
        genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"다음 문서를 참고하여 질문에 답하세요:\n{context}\n\n질문: {query}",
        )
        print("🧠 Agent Response:\n", response.text)
    except Exception as e:
        print("⚠️ Google GenAI error:", e)
        print("🔍 Fallback Answer:")
        print(context[:1000])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build ChromaDB index from PDFs")
    parser.add_argument("--chat", type=str, help="Ask a question to the RAG agent", default=None)
    args = parser.parse_args()

    if args.build:
        build_vector_store()
    elif args.chat:
        chat_with_agent(args.chat)
    else:
        print("⚙️ Usage: python rag_agent_chroma_adk.py --build | --chat '질문 내용'")


# python rag_agent_chroma_adk.py --build : "PDF 임베딩 생성"
# python rag_agent_chroma_adk.py --chat '질문 내용 입력' : "PDF 내용 관련 질문"

# 다음 단계,
# 임베딩 생성 코드와 챗 부분을 분리
# 챗 부분은 Agent로 생성하여 장문의 template를 입력할 수 있도록 수정 (251014)
