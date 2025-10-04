# 긍정/부정에 대한 서브 에이전트를 호출하는 구조

# 별도의 파일로 관리되는 서브 에이전트 호출 방법 이해
# 별도의 파일로 관리되는 instruction 호출 방법 이해

from google.adk.agents import Agent
# from google.adk.tools import google_search
from .sub_agent import positive_critic, negative_critic

# ADK에는 google search, rag engine, verteai search 툴이 내장되어 있다.

from . import instruction     # 별도의 파일로 저장한 instruction을 import 하는 방법

def build_agent() -> Agent:   # 출력 형식이 Agent인 메서드 정의  -> 왜 이렇게 작성하는가? (251004)
    """
    비평 작업을 위한 서브 에이전트를 포함하는 루트 Agent 인스턴스를 생성하는 설정함수입니다.
    - 환경 변수를 불러오고, 에이전트의 지시문 템플릿을 정의합니다.
    - 이 에이전트는 자체 지식과 검색 기능을 모두 활용하여 사용자의 문의에 답하도록 설계되었습니다.
    - 긍정 및 부정 비평을 위한 서브 에이전트 설정합니다.
    - 사용자의 요청에 따라 특정 비평 작업을 위임합니다.

    반환값:
    Agent: 사용자 질의 처리가 가능한 서브 에이전트가 포함된 설정된 Agent 인스턴스 반환 """

# - Google Search 도구 지원이 포함된 Agent 인스턴스를 생성하고 구성합니다.
# - 이름, 모델, 설명, 안내문, Google Search 도구를 포함하여 Agent를 초기화합니다.


    agent = Agent(
        name = "root_agent",
        model = "gemini-2.5-flash",
        description = "사용자의 질의에 답변하는 에이전트",
        instruction = instruction.INSTRUCTION,
        # tools = [google_search],
        sub_agents = [positive_critic, negative_critic],
    )
    return agent

root_agent = build_agent()