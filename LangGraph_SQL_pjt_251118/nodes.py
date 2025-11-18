from langchain_openai import ChatOpenAI
from faiss_manager import add_to_faiss, search_similar

def get_llm():
    return ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="lm-studio",
        model="lmstudio-community/gemma-3-4b-it-GGUFF",
        temperature=0.1,
    )

llm = get_llm()

def retrieve_context_node(state):
    user_input = state.get("user_input", "")
    best_match = search_similar(user_input)
    return {"history": best_match}  # ✅ OK

def chatbot_node(state):
    user_input = state.get("user_input", "")
    history = state.get("history", "")
    prompt = f"""당신은 친절한 챗봇입니다.
    과거 대화: {history}
    현재 사용자 입력: {user_input}
    자연스럽고 간결하게 응답하세요.
    """
    print(history)   # 매칭된 결과를 확인하여 대화 기록을 제대로 탐색했는지 확인 (251118)
    
    # ⚠️ predict()는 deprecated, invoke() 사용 권장
    # bot_response = llm.predict(prompt)
    bot_response = llm.invoke(prompt).content  # 수정
    
    add_to_faiss(user_input, bot_response)
    return {"message": bot_response}  # user_input은 유지됨