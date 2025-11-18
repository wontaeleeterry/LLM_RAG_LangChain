import faiss
import os
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

# HuggingFace 임베딩 모델 설정
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  # 다국어 모델
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

# 이 모델의 실제 차원은 384 (1536이 아님!)
dimension = 384

index_file = "faiss_index.bin"
stored_texts = []

# FAISS 인덱스 초기화
if os.path.exists(index_file):
    try:
        index = faiss.read_index(index_file)
        print(f"기존 FAISS 인덱스 로드: {index.ntotal}개 벡터")
    except Exception as e:
        print(f"기존 인덱스 로드 실패: {e}")
        index = faiss.IndexFlatL2(dimension)
else:
    index = faiss.IndexFlatL2(dimension)
    print("새 FAISS 인덱스 생성")

def add_to_faiss(user_input: str, bot_response: str):
    """대화를 FAISS 인덱스에 추가"""
    combined_text = f"User: {user_input} | Bot: {bot_response}"
    
    # 임베딩 생성
    vector = embedding_model.embed_query(combined_text)
    
    # numpy 배열로 변환 (shape: (1, dimension))
    vector_array = np.array([vector], dtype="float32")
    
    # 차원 검증
    if vector_array.shape[1] != dimension:
        print(f"경고: 벡터 차원 불일치 - 예상: {dimension}, 실제: {vector_array.shape[1]}")
        return
    
    # FAISS에 추가
    index.add(vector_array)
    stored_texts.append(combined_text)
    
    # 인덱스 저장
    faiss.write_index(index, index_file)
    print(f"FAISS에 추가됨: 총 {len(stored_texts)}개 대화")

# 유사도 결과를 여러 개 가져오기 (251118)
def search_similar(query: str, top_k: int = 3) -> str:  # top_k: int = 1 (251118)
    """유사한 과거 대화 검색"""
    if len(stored_texts) == 0:
        return ""
    
    # 쿼리 임베딩 생성
    query_vector = embedding_model.embed_query(query)
    query_array = np.array([query_vector], dtype="float32")
    
    # 차원 검증
    if query_array.shape[1] != dimension:
        print(f"경고: 쿼리 벡터 차원 불일치")
        return ""
    
    # FAISS 검색
    distances, indices = index.search(query_array, min(top_k, len(stored_texts)))
    
    # 결과 수집
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if 0 <= idx < len(stored_texts):
            results.append(stored_texts[idx])
    
    return "\n".join(results) if results else ""

# 테스트: 임베딩 차원 확인
# test_vector = embedding_model.embed_query("테스트")
# print(f"임베딩 모델 실제 차원: {len(test_vector)}")