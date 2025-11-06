# multi_agent_graph.py
from typing import TypedDict
from langgraph.graph import StateGraph, END
from sub_agent import get_llm
from instruction import summary_prompt, translation_prompt, review_prompt


# -------------------------------
# 상태 정의 (LangGraph의 데이터 흐름 구조)
# -------------------------------
class WorkflowState(TypedDict, total=False):
    """각 단계 결과를 저장하는 데이터 컨테이너"""
    input: str
    summary: str
    translation: str
    final: str


# -------------------------------
# 노드 정의
# -------------------------------
def summarize_node(state: WorkflowState):
    llm = get_llm()
    print("🧩 요약 중...")
    result = llm.invoke(f"{summary_prompt}\n\n{state['input']}")
    print(result)
    return {"summary": result.content}


def translate_node(state: WorkflowState):
    llm = get_llm()
    print("🌍 번역 중...")
    result = llm.invoke(f"{translation_prompt}\n\n{state['summary']}")
    print(result)
    return {"translation": result.content}


def review_node(state: WorkflowState):
    llm = get_llm()
    print("🔍 검토 중...")
    result = llm.invoke(f"{review_prompt}\n\n{state['translation']}")
    print(result)
    print("✅ 모든 단계 완료!\n")
    return {"final": result.content}


# -------------------------------
# 그래프 구성
# -------------------------------
def build_workflow_graph():
    workflow = StateGraph(WorkflowState)

    workflow.add_node("summary", summarize_node)
    workflow.add_node("translation", translate_node)
    workflow.add_node("review", review_node)

    workflow.set_entry_point("summary")
    workflow.add_edge("summary", "translation")
    workflow.add_edge("translation", "review")
    workflow.add_edge("review", END)

    return workflow.compile()