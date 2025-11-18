from graph_builder import build_graph

def run_chatbot():
    graph = build_graph()
    thread_id = "chat_session_1"
    step_count = 1
    print("Chatbot 시작 : 종료하려면 'exit' 또는 'q' 입력")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "q"]:
            print("종료합니다.")
            break
        
        state = {"user_input": user_input, "history": "", "message": ""}
        config = {"configurable": {"thread_id": thread_id}}  # config 형식 수정
        
        result = graph.invoke(state, config=config)
        step_count += 1
        
        print("=== 대화 상태 ===")
        if "message" in result:
            print(f"Bot: {result['message']}")
        print("=================\n")

if __name__ == "__main__":
    run_chatbot()