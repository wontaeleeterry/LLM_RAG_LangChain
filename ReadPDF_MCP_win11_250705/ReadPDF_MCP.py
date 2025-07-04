from mcp.server.fastmcp import FastMCP
import traceback
from sentence_transformers import SentenceTransformer
import os
import chromadb
import numpy as np
import re
from pathlib import Path
import platform

# pdfminer.six 관련 import
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument, PDFEncryptionError
from io import StringIO

mcp = FastMCP("ReadPDF_MCP")

# 임베딩 모델
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(chunks):
    return model.encode(chunks)

# 벡터DB (ChromaDB)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("pdf_chunks")

def build_or_load_index(chunks):
    if not chunks:
        return collection, []
    
    vectors = get_embeddings(chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    try:
        chroma_client.delete_collection("pdf_chunks")
        collection = chroma_client.create_collection("pdf_chunks")
    except:
        pass
    
    collection.add(documents=chunks, embeddings=vectors, ids=ids)
    return collection, chunks

def search_similar_chunks(query, collection, chunks, top_k=5):
    if not chunks:
        return []
    
    q_vec = get_embeddings([query])[0]
    results = collection.query(query_embeddings=[q_vec], n_results=min(top_k, len(chunks)))
    return results['documents'][0] if results['documents'] else []

def extract_text_from_pdf(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"[오류] 파일이 존재하지 않습니다: {file_path}")
            return ""

        try:
            text = extract_text(file_path)
            if text and text.strip():
                return text
        except PDFEncryptionError:
            print(f"[오류] PDF가 암호화되어 있습니다: {file_path}")
            return ""
        except Exception as e:
            print(f"간단한 추출 실패, 고급 추출 시도: {e}")

        try:
            output_string = StringIO()
            with open(file_path, 'rb') as file:
                laparams = LAParams(line_margin=0.5, word_margin=0.1, char_margin=2.0)
                resource_manager = PDFResourceManager()
                device = TextConverter(resource_manager, output_string, laparams=laparams)
                interpreter = PDFPageInterpreter(resource_manager, device)

                for page in PDFPage.get_pages(file, check_extractable=True):
                    interpreter.process_page(page)

                device.close()
                text = output_string.getvalue()
                output_string.close()

                return text
        except Exception as e:
            print(f"고급 추출도 실패: {e}")
            return ""
    except Exception as e:
        print(f"PDF 읽기 오류 ({file_path}): {e}")
        return ""

def split_into_chunks(text, chunk_size=300):
    if not text.strip():
        return []

    sentences = re.split(r'[.!?]\s+', text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

        if len(potential_chunk) <= chunk_size:
            current_chunk = potential_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(sentence) <= chunk_size:
                current_chunk = sentence
            else:
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk + " " + word) <= chunk_size:
                        temp_chunk = temp_chunk + " " + word if temp_chunk else word
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word
                current_chunk = temp_chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if chunk.strip()]

def find_pdf_files(search_dirs=None, specific_path=None):
    pdf_files = []

    if search_dirs is None:
        if platform.system() == "Windows":
            default_dir = os.path.join(os.path.expanduser("~"), "Desktop", "filesystem_MCP")
        else:
            default_dir = "/Users/wonta/Desktop/filesystem_MCP/"

        search_dirs = [
            default_dir,
            os.path.join(os.getcwd(), "docs"),
            "docs/", "./", "../", "uploads/"
        ]

    for dir_path in search_dirs:
        if not os.path.isabs(dir_path):
            dir_path = os.path.abspath(dir_path)

        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            try:
                for file in os.listdir(dir_path):
                    if file.lower().endswith(".pdf"):
                        full_path = os.path.join(dir_path, file)
                        pdf_files.append(os.path.abspath(full_path))
            except PermissionError:
                continue
    return pdf_files

def find_pdf_files_recursive(root_path):
    pdf_files = []
    try:
        if not os.path.exists(root_path):
            return pdf_files

        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    full_path = os.path.join(root, file)
                    pdf_files.append(os.path.abspath(full_path))
    except Exception as e:
        print(f"재귀 검색 중 오류: {e}")
    return pdf_files

def load_and_chunk_documents(search_dirs=None, specific_path=None):
    pdf_files = find_pdf_files(search_dirs, specific_path)

    if not pdf_files and not specific_path:
        print("기본 디렉토리에서 PDF를 찾을 수 없어 재귀 검색을 시도합니다...")

        if platform.system() == "Windows":
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "filesystem_MCP")
        else:
            desktop_path = "/Users/wonta/Desktop/filesystem_MCP/"

        search_paths = [desktop_path, os.getcwd()]

        for search_path in search_paths:
            if os.path.exists(search_path):
                pdf_files = find_pdf_files_recursive(search_path)
                if pdf_files:
                    break

    if not pdf_files:
        print("PDF 파일을 찾을 수 없습니다.")
        return []

    all_chunks = []
    for file_path in pdf_files:
        print(f"처리 중: {file_path}")
        raw_text = extract_text_from_pdf(file_path)
        if raw_text.strip():
            chunks = split_into_chunks(raw_text)
            all_chunks.extend(chunks)
            print(f"  - {len(chunks)}개 청크 생성")
        else:
            print(f"  - 텍스트 추출 실패")

    print(f"총 {len(all_chunks)}개 청크 생성됨")
    return all_chunks

@mcp.tool(name="query_documents", description="PDF 문서를 검색하고 Claude에 질문을 전달합니다.")
def query_documents(query: str) -> str:
    try:
        print(f"[질문] {query}")
        print(f"[현재 작업 디렉토리] {os.getcwd()}")

        docs = load_and_chunk_documents()

        if not docs:
            return "오류 발생: PDF 파일을 찾을 수 없습니다."

        collection, chunks = build_or_load_index(docs)
        results = search_similar_chunks(query, collection, chunks)

        if not results:
            return "관련 문서를 찾을 수 없습니다."

        prompt_response = "\n".join([
            "다음 문단을 참고하여 질문에 답해주세요:\n",
            *[f"- {chunk}" for chunk in results],
            f"\n질문: {query}"
        ])
        return prompt_response

    except Exception as e:
        error_msg = f"오류 발생: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg

if __name__ == "__main__":
    mcp.run()
