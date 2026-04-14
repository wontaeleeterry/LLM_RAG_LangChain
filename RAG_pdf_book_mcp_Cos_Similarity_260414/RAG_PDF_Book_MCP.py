from fastmcp import FastMCP
from RAG_search import RAGEngine


mcp = FastMCP("RAG PDF Book MCP")
rag = RAGEngine()
rag.load_db()  # 서버 시작 시 1회 preload


@mcp.tool()
def retrieve_context(query: str, top_k: int = 10) -> str:
    """
    사용자 질문(query)에 대해 벡터 유사도 기반 상위 context를 반환합니다.
    Claude는 반환된 context를 기반으로 최종 답변을 생성합니다.
    """
    #context = rag.build_context(query, top_k=top_k)
    context = rag.ask(query, top_k=top_k)

    return f"""다음은 사용자 질문과 가장 유사한 문서 context입니다.

질문:
{query}

검색된 Context (Top {top_k}):
{context}

위 context만 근거로 정확하게 답변하세요.
"""


if __name__ == "__main__":
    mcp.run()