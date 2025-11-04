"""
다국어 콘텐츠 번역 및 검토 워크플로우 멀티 에이전트 시스템
Agent 정의 파일
"""

# from google.adk.agents import Agent
# from google.adk.tools import FunctionTool

# --- ADK 환경이 없을 때를 위한 임시 클래스 정의 ---
'''
이 코드는 ADK가 설치되지 않은 로컬 환경에서도 프로그램이 멈추지 않고 작동하도록, 
“ADK의 핵심 클래스들을 간단히 흉내 내는 가짜 버전”을 자동 생성하는 코드입니다.
'''

try:
    from google.adk.agents import Agent
    from google.adk.tools import FunctionTool, ToolContext

except ModuleNotFoundError:
    print("⚠️ google.adk 모듈을 찾을 수 없습니다. 로컬 시뮬레이션용 가짜 클래스를 사용합니다.")

    class FunctionTool:
        def __init__(self, func):
            self.func = func

    class ToolContext:
        def __init__(self):
            self.session = {}

    class Agent:
        def __init__(self, name, model, description, instruction, tools):
            self.name = name
            self.model = model
            self.description = description
            self.instruction = instruction
            self.tools = tools

        def run(self, *args, **kwargs):
            print(f"[{self.name}] is running (simulated).")
            if self.tools:
                return self.tools[0].func(ToolContext(), *args, **kwargs)
            return {"success": False, "error": "No tool attached"}


from .instruction import (
    DOCUMENT_LOADER_INSTRUCTION,
    SUMMARY_EXPERT_INSTRUCTION,
    TRANSLATION_EXPERT_INSTRUCTION,
    QUALITY_REVIEW_EXPERT_INSTRUCTION
)
from .sub_agent import (
    load_document_tool,
    summarize_content_tool,
    translate_content_tool,
    review_translation_tool
)


# 1. 문서 로더 에이전트
document_loader_agent = Agent(
    name="document_loader_agent",
    model="gemini-2.0-flash",
    description="원본 문서를 파일에서 읽어오는 전문 에이전트. 'original_document' 키로 세션에 저장합니다.",
    instruction=DOCUMENT_LOADER_INSTRUCTION,
    tools=[FunctionTool(load_document_tool)]
)


# 2. 요약 전문가 에이전트
summary_expert_agent = Agent(
    name="summary_expert_agent",
    model="gemini-2.0-flash",
    description="원본 문서를 받아 핵심 내용을 추출하여 요약하는 전문 에이전트. 'summary' 키로 세션에 저장합니다.",
    instruction=SUMMARY_EXPERT_INSTRUCTION,
    tools=[FunctionTool(summarize_content_tool)]
)


# 3. 번역 전문가 에이전트
translation_expert_agent = Agent(
    name="translation_expert_agent",
    model="gemini-2.0-flash",
    description="요약된 내용을 대상 언어로 번역하는 전문 에이전트. 'translation' 키로 세션에 저장합니다. target_language 파라미터를 받습니다.",
    instruction=TRANSLATION_EXPERT_INSTRUCTION,
    tools=[FunctionTool(translate_content_tool)]
)


# 4. 품질 검토 전문가 에이전트
quality_review_expert_agent = Agent(
    name="quality_review_expert_agent",
    model="gemini-2.0-flash",
    description="번역 결과를 검토하여 문법, 맥락, 전문 용어 사용의 정확성을 평가하고 수정하는 전문 에이전트. 'final_translation' 키로 세션에 저장합니다.",
    instruction=QUALITY_REVIEW_EXPERT_INSTRUCTION,
    tools=[FunctionTool(review_translation_tool)]
)


# 모든 에이전트를 포함한 리스트
all_agents = [
    document_loader_agent,
    summary_expert_agent,
    translation_expert_agent,
    quality_review_expert_agent
]

# --- 루트 에이전트 정의 (로컬 시뮬레이션용) ---
'''
이 코드는 ADK가 설치되지 않은 로컬 환경에서도 프로그램이 멈추지 않고 작동하도록, 
“ADK의 핵심 클래스들을 간단히 흉내 내는 가짜 버전”을 자동 생성하는 코드입니다.
'''

try:
    from google.adk.agents import Agent

except Exception:
    # Stub Fallback
    class Agent:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def run(self, *args, **kwargs):
            print("[root_agent] Simulated run (no external network calls).")
            return {"success": True, "message": "Simulated root agent run."}

root_agent = Agent(
    name="root_agent",
    model="gemini-2.5-flash",
    # model=None,  # ⚠️ 외부 모델 호출 방지
    description="문서 요약, 번역, 품질 검토를 순차적으로 수행하는 로컬 시뮬레이터",
    instruction=(
        "이 시스템은 실제 네트워크 호출 없이 로컬 서브 에이전트를 순차 실행합니다."
    ),
    tools=[]
)
