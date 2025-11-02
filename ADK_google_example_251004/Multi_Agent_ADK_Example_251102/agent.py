from google.adk.agents import Agent
from .sub_agent import data_loader_agent, data_cleaner_agent, data_analyzer_agent, data_visualizer_agent
from . import instruction

def build_agent() -> Agent:
    """
    Root Agent 생성 함수.
    - 데이터 분석 파이프라인을 전체적으로 조정
    - Sub-Agent를 호출하여 각 단계 작업 위임
    """
    agent = Agent(
        name="root_agent",
        model="gemini-2.0-flash",
        description="사용자의 데이터 분석 요청을 처리하는 루트 에이전트",
        instruction=instruction.INSTRUCTION,
        sub_agents=[data_loader_agent,
                    data_cleaner_agent,
                    data_analyzer_agent,
                    data_visualizer_agent],
    )
    return agent

root_agent = build_agent()