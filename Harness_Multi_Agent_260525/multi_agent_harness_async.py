"""
멀티 에이전트 하네스 예제 (병렬 처리 버전)
──────────────────────────────────────────
asyncio를 활용하여 3개 에이전트를 동시에 실행하고
하네스 에이전트가 결과를 종합합니다.

단일 실행 대비 처리 속도가 최대 3배 빠릅니다.
"""

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import StreamingStdOutCallbackHandler


# ─────────────────────────────────────────────
# LLM 팩토리
# ─────────────────────────────────────────────
def make_llm(streaming: bool = False) -> ChatOpenAI:
    callbacks = [StreamingStdOutCallbackHandler()] if streaming else []
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="lmstudio-community/gemma-2-2b-it-GGUF",
        temperature=0.1,
        streaming=streaming,
        callbacks=callbacks,
    )


# ─────────────────────────────────────────────
# 에이전트 체인 빌더 (경량화된 함수형 스타일)
# ─────────────────────────────────────────────
def build_agent_chain(system_prompt: str, human_template: str):
    """프롬프트 + LLM + 파서로 구성된 체인을 반환합니다."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template),
    ])
    return prompt | make_llm() | StrOutputParser()


# 리서처 체인
researcher_chain = build_agent_chain(
    system_prompt=(
        "당신은 정보 수집 전문가입니다. "
        "주어진 주제에 대해 핵심 사실, 배경, 관련 맥락을 "
        "3~5개의 불릿 포인트로 간결하게 정리하세요. 한국어로 답하세요."
    ),
    human_template="다음 주제를 조사해 주세요:\n\n{topic}",
)

# 분석가 체인
analyst_chain = build_agent_chain(
    system_prompt=(
        "당신은 데이터 분석 전문가입니다. "
        "주어진 주제에 대해 원인, 영향, 핵심 요인을 논리적으로 분석하여 "
        "3~5개의 분석 포인트로 정리하세요. 한국어로 답하세요."
    ),
    human_template="다음 주제를 분석해 주세요:\n\n{topic}",
)

# 전략가 체인
strategist_chain = build_agent_chain(
    system_prompt=(
        "당신은 전략 기획 전문가입니다. "
        "주어진 주제에 대해 즉시 실행 가능한 구체적인 전략이나 권고사항을 "
        "3~5개의 액션 아이템으로 제시하세요. 한국어로 답하세요."
    ),
    human_template="다음 주제에 대한 전략을 제시해 주세요:\n\n{topic}",
)

# 하네스 체인 (스트리밍)
harness_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 여러 전문가의 의견을 종합하는 수석 컨설턴트입니다. "
     "리서처의 배경 조사, 분석가의 분석 결과, 전략가의 제안을 통합하여 "
     "일관성 있는 최종 보고서를 작성하세요. "
     "보고서는 ① 핵심 요약, ② 주요 발견, ③ 권고사항 섹션으로 구성하세요. "
     "항상 한국어로 답하세요."),
    ("human",
     "주제: {topic}\n\n"
     "─── 리서처 조사 결과 ───\n{research}\n\n"
     "─── 분석가 분석 결과 ───\n{analysis}\n\n"
     "─── 전략가 제안 ───\n{strategy}\n\n"
     "위 세 전문가의 내용을 종합하여 최종 보고서를 작성해 주세요."),
])
harness_chain = harness_prompt | make_llm(streaming=True) | StrOutputParser()


# ─────────────────────────────────────────────
# 병렬 파이프라인
# ─────────────────────────────────────────────
async def run_agents_parallel(topic: str) -> tuple[str, str, str]:
    """3개 에이전트를 비동기로 동시 실행합니다."""
    print("\n[3개 에이전트 병렬 실행 중...]\n")

    research_task  = researcher_chain.ainvoke({"topic": topic})
    analysis_task  = analyst_chain.ainvoke({"topic": topic})
    strategy_task  = strategist_chain.ainvoke({"topic": topic})

    research, analysis, strategy = await asyncio.gather(
        research_task, analysis_task, strategy_task
    )
    return research, analysis, strategy


async def run_pipeline_async(topic: str) -> str:
    print("\n" + "━" * 60)
    print(f"  주제: {topic}")
    print("━" * 60)

    # ── Step 1: 병렬 실행 ──────────────────────────────────────
    research, analysis, strategy = await run_agents_parallel(topic)

    # 중간 결과 출력
    for label, content in [
        ("리서처 결과", research),
        ("분석가 결과", analysis),
        ("전략가 결과", strategy),
    ]:
        print(f"\n{'─' * 60}\n  {label}\n{'─' * 60}")
        print(content)

    # ── Step 2: 하네스 종합 ────────────────────────────────────
    print("\n[하네스 에이전트: 종합 보고서 생성 중...]\n")
    print("=" * 60)

    final = await harness_chain.ainvoke({
        "topic":    topic,
        "research": research,
        "analysis": analysis,
        "strategy": strategy,
    })

    print("\n" + "=" * 60)
    return final


# ─────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────
if __name__ == "__main__":
    TOPIC = "중소기업의 AI 도입 전략"

    final_report = asyncio.run(run_pipeline_async(TOPIC))

    with open("final_report_async.txt", "w", encoding="utf-8") as f:
        f.write(f"주제: {TOPIC}\n\n{final_report}")
    print("\n→ final_report_async.txt 에 저장되었습니다.")
