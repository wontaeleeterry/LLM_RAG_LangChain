from google.adk.agents import Agent

positive_critic = Agent(
    name = "positive_critic",
    model = "gemini-2.5-flash",
    description = "사용자의 질문에 긍정적인 측면만 답변하는 에이전트.",
    instruction = """당신은 사용자가 문의한 질문에 긍정적인 리뷰를 작성하는 에이전트입니다.
                  답변을 제공할 때는 최대한 간결하고 명확하게 작성하며, 
                  반드시 '긍정적 리뷰 결과:' 라는 말로 시작하세요.""",
)

negative_critic = Agent(
    name = "negative_critic",
    model = "gemini-2.5-flash",
    description = "사용자의 질문에 부정적인 측면만 답변하는 에이전트.",
    instruction = """당신은 사용자가 문의한 질문에 부정적인 리뷰를 작성하는 에이전트입니다.
                  답변을 제공할 때는 최대한 간결하고 명확하게 작성하면,
                  반드시 '부정적 리뷰 결과:' 라는 말로 시작하세요."""
)
