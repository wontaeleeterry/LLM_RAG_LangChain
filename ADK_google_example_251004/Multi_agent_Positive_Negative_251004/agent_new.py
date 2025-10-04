# 기본 에이전트
# 별도의 파일로 관리되는 서브 에이전트 호출 방법 이해
# 별도의 파일로 관리되는 instruction 호출 방법 이해

from google.adk.agents import Agent
from google.adk.tools import google_search

# ADK에는 google search, rag engine, verteai search 툴이 내장되어 있다.

from . import instruction     # 별도의 파일로 저장한 instruction을 import 하는 방법

def build_agent() -> Agent:   # 출력 형식이 Agent인 메서드 정의  -> 왜 이렇게 작성하는가? (251004)
    """
    Google Search 도구 지원이 포함된 Agent 인스턴스를 생성하고 구성합니다.
    이 함수는 환경 변수를 로드하고, 에이전트의 안내 템플릿을 설정하며,
    이름, 모델, 설명, 안내문, Google Search 도구를 포함하여 Agent를 초기화합니다.
    이 에이전트는 자체 지식과 검색 기능을 모두 활용하여 사용자의 문의에 답하도록 설계되었습니다.
    반환값:
    Agent: 사용자 질의 처리가 가능한 구성된 Agent 인스턴스 """

    agent = Agent(
        name = "search_agent",
        model = "gemini-2.0-flash",
        description = "사용자의 질의에 답변하는 에이전트",
        instruction = instruction.INSTRUCTION,
        tools = [google_search],
    )
    return agent

root_agent = build_agent()