from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from nodes import retrieve_context_node, chatbot_node
from typing import TypedDict
import sqlite3

class ChatState(TypedDict):
    user_input: str
    history: str
    message: str

def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("retrieve_context_node", retrieve_context_node)
    builder.add_node("chatbot_node", chatbot_node)
    builder.set_entry_point("retrieve_context_node")
    builder.add_edge("retrieve_context_node", "chatbot_node")
    builder.set_finish_point("chatbot_node")
    
    # SqliteSaver가 올바른 스키마로 테이블을 자동 생성
    conn = sqlite3.connect("conversation.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()  # ← 중요: 테이블 스키마 생성
    
    return builder.compile(checkpointer=checkpointer)