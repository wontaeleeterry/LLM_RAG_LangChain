# sub_agent.py
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools


def get_llm():
    """LangGraph에서 사용할 LLM 설정"""
    return ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="lm-studio",
        model="lmstudio-community/gemma-3-4b-it-GGUFF",
        temperature=0.1,
    )


def build_agent():
    """LangGraph용 기본 Agent 생성"""
    llm = get_llm()
    tools = load_tools([], llm=llm)
    return create_react_agent(model=llm, tools=tools)


def load_doc():
    with open('./doc.txt', 'r', encoding='utf-8') as f:
        return f.read()
