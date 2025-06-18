```python
# 참고 : https://velog.io/@tetrapod0/LLM-%EC%BD%94%EB%93%9C%EC%97%90%EC%84%9C-%EB%8F%8C%EB%A0%A4%EB%B3%B4%EA%B8%B0


from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="lm-studio",
    model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()], # 스트림 출력 콜백
)
```


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "{input} 한국어로 답변해줘."
)

chain = prompt | llm | StrOutputParser()


response = chain.invoke("안녕!")
# response = chain.invoke({'input' : "안녕!"})
```

    안녕하세요! 한국어로 답변해드리겠습니다.
    
    무엇에 대해 질문하실 건가요?


```python
# 디렉토리 내 모든 파일을 리스트로 변환하는 함수 정의

import os

def list_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

# 지정된 디렉토리 내 모든 파일명을 리스트로 호출
file_names = list_files('./data')
print(file_names)
```

    ['./data\\constitution_of_Korea.pdf']
    


```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 문장을 임베딩으로 변환하고 벡터 저장소에 저장
embeddings_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',    # 다국어 모델
    # model_name='jhgan/ko-sroberta-multitask',  # 한국어 모델 - 에러 발생 (250603)
    # model_name = 'BAAI/bge-m3',                # 에러 발생 (250603)
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)

embeddings_model
```

    C:\Users\wonta\AppData\Local\Temp\ipykernel_8496\4240770564.py:8: LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in LangChain 0.2.2 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-huggingface package and should be used instead. To use it run `pip install -U :class:`~langchain-huggingface` and import as `from :class:`~langchain_huggingface import HuggingFaceEmbeddings``.
      embeddings_model = HuggingFaceEmbeddings(
    

    WARNING:tensorflow:From c:\Users\wonta\anaconda3\Lib\site-packages\tf_keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.
    
    




    HuggingFaceEmbeddings(client=SentenceTransformer(
      (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: BertModel 
      (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
    ), model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_folder=None, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True}, multi_process=False, show_progress=False)




```python
loader = PyMuPDFLoader(file_names[0])       # 폴더 내 파일 1개만 존재 : 여러 개일 경우, 최초 1개 DB 생성 후, Add 방식으로 진행 (250605)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=16) 
docs = text_splitter.split_documents(documents)

# 임베딩 DB 생성 : 파일로 저장하지 않으므로, 새로 실행할 경우 초기화됨 (250605)
db_constitution = Chroma.from_documents(
    documents=docs, embedding=embeddings_model, collection_name="db_constitution"
)
```


```python
# db_constitution.similarity_search("대통령의 권한과 의무에 대한 내용", k=20)

# 헌법 전문을 모두 참고하는 경우,
retriever_all = db_constitution.as_retriever()

# 특정 내용을 검색하고 그 결과를 참고하는 경우,

# 검색 쿼리
query = '대통령'    # 키워드에 대한 내용을 먼저 추출

# 가장 유사도가 높은 문장 추출
retriever = db_constitution.as_retriever(search_kwargs={'k': 20})
docs = retriever.get_relevant_documents(query)

```


```python
len(docs)
```




    20




```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# Prompt 템플릿 생성
template = '''Answer the question based only on the following context:
{context} Please answer all the answers in Korean.:

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

# RAG Chain 연결
rag_chain = prompt | llm | StrOutputParser()

# Chain 실행
query = "대통령은 탄핵될 수 있나요?"
answer = rag_chain.invoke({'context': (format_docs(docs)), 'question': query}) 
# 기 출출된 20개의 context 범위 내에서 question에 대한 응답을 찾을 것
```

    헌법 제67조에 따르면, 대통령은 국민의 보통ㆍ평등ㆍ직접ㆍ비밀선거에 의하여 선출된다. 그러나 헌법 제83조에는 "탄핵소추의 의결을 받은 자는 탄핵심판이 있을 때까지 그 권한행사가 정지된다."고 명시되어 있습니다.
    
    따라서, 대통령은 탄핵될 수 있지만, 탄핵소추가 이루어진 경우에는 권한행사가 정지됩니다.


```python
# Chain 실행
query = "국민의 의무에 관하여 남성과 여성의 차이가 있나요?"
answer = rag_chain.invoke({'context': retriever, 'question': query}) 
# answer = rag_chain.invoke(query)
```

    기본적으로 국민의 의무는 성별에 관계없이 동일합니다. 그러나 일부 법적 의무가 성별에 따라 다를 수 있습니다.
    
    예를 들어, 병역 의무는 남성에게만 적용되며, 여성은 병역 비과세 혜택을 받습니다. 또한, 가족법상 부양의무도 일반적으로 아버지나 어머니 중 한 명에게 부여됩니다.
    
    그러나 이러한 차이는 법률에 의해 정해져 있으며, 성별에 따른 차이가 있는 법적 의무는 매우 제한적입니다.


```python
# 추출한 내용 중에 질문에 관한 것이 없는 경우,
# docs : '대통령'에 대한 내용만 포함되어 있다.

query = "국민의 의무에 관하여 남성과 여성의 차이가 있나요?"
answer = rag_chain.invoke({'context': (format_docs(docs)), 'question': query}) 
```

    해당되는 문구는 없습니다. 따라서, 답변할 수 없습니다.


```python

```
