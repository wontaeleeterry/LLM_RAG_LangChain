from crewai import Agent, Task, Crew, LLM

llm = LLM(
    model="openai/lmstudio-community/qwen/qwen3-coder-30b",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    temperature=0.1,
)

research_agent = Agent(
    role="Researcher",
    goal="질문을 분석한다.",
    backstory="분석 전문가",
    llm=llm
)

task = Task(
    description="CrewAI가 무엇인지 설명하라.",
    expected_output="설명 문서",
    agent=research_agent
)

crew = Crew(
    agents=[research_agent],
    tasks=[task]
)

result = crew.kickoff()

print(result)

'''
멀티 에이전트 프레임워크를 이용할 경우,
답변 품질이 월등히 향상되었는지 확인해볼 것 (260630)
'''