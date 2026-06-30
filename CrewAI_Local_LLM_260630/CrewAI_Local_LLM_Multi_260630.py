from crewai import Agent, Task, Crew, LLM

llm = LLM(
    model="openai/lmstudio-community/qwen/qwen3-coder-30b",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    temperature=0.1,
)

####################################################
# Agent 정의
####################################################

research_agent = Agent(
    role="Research Engineer",
    goal="사용자의 요구사항을 분석하고 필요한 정보를 수집한다.",
    backstory="""
    당신은 뛰어난 요구사항 분석가이다.
    문제를 작은 단위로 나누고 핵심 기능을 정의한다.
    """,
    llm=llm,
    verbose=True
)

architect_agent = Agent(
    role="Software Architect",
    goal="시스템 구조와 구현 전략을 설계한다.",
    backstory="""
    당신은 시니어 소프트웨어 아키텍트이다.
    확장 가능하고 유지보수하기 쉬운 구조를 설계한다.
    """,
    llm=llm,
    verbose=True
)

developer_agent = Agent(
    role="Senior Python Developer",
    goal="설계에 따라 파이썬 코드를 구현한다.",
    backstory="""
    당신은 10년 경력의 Python 개발자이다.
    읽기 쉽고 테스트 가능한 코드를 작성한다.
    """,
    llm=llm,
    verbose=True
)

reviewer_agent = Agent(
    role="Code Reviewer",
    goal="코드 품질과 버그를 검토한다.",
    backstory="""
    당신은 코드 리뷰 전문가이다.
    성능, 보안, 유지보수성을 검토한다.
    """,
    llm=llm,
    verbose=True
)

####################################################
# Task 정의
####################################################

task1 = Task(
    description="""
    사용자의 요구사항:
    'SQLite를 사용하는 간단한 Todo 관리 프로그램을 만들어라.'

    필요한 기능을 분석하고
    핵심 요구사항을 정리하라.
    """,
    expected_output="""
    기능 목록과 요구사항 명세서
    """,
    agent=research_agent
)

task2 = Task(
    description="""
    이전 분석 결과를 바탕으로
    클래스 구조와 데이터베이스 설계를 작성하라.
    """,
    expected_output="""
    시스템 설계서
    """,
    agent=architect_agent
)

task3 = Task(
    description="""
    설계서를 기반으로
    SQLite Todo 프로그램을 구현하라.
    """,
    expected_output="""
    실행 가능한 Python 코드
    """,
    agent=developer_agent
)

task4 = Task(
    description="""
    생성된 코드를 리뷰하고
    버그와 개선점을 제안하라.
    """,
    expected_output="""
    코드 리뷰 보고서
    """,
    agent=reviewer_agent
)

####################################################
# Crew 구성
####################################################
from crewai import Process


crew = Crew(
    agents=[
        research_agent,
        architect_agent,
        developer_agent,
        reviewer_agent
    ],
    tasks=[
        task1,
        task2,
        task3,
        task4
    ],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()

print(result)
