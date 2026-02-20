import sys
from ddgs import DDGS
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("web-search-mcp")

def log(msg: str):
    """LM Studio stderr 로그에 출력"""
    print(f"[web-search] {msg}", file=sys.stderr, flush=True)

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Perform a stable and optimized web search using DuckDuckGo.
    """
    log(f" 검색 시작: '{query}'")

    # 1 입력 검증
    if not isinstance(query, str) or not query.strip():
        return "Invalid query."
    try:
        max_results = int(max_results)
    except:
        max_results = 5
    max_results = max(1, min(max_results, 15))

    # 2 원본 쿼리 저장 후 region 판별
    original_query = query.strip()
    # DDGS 지역 코드는 보통 'kr-kr' 보다는 'kr-kr' 혹은 'wt-wt' 형식을 쓰지만 
    # 최신 버전 호환성을 위해 소문자 유지
    region = "us-en" if original_query.isascii() else "kr-kr"
    log(f" region: {region}, max_results: {max_results}")

    # 3 원치 않는 사이트 차단
    blocked_sites = ["baidu.com", "zhidao.baidu.com", "so.com", "sogou.com"]
    filtered_query = original_query
    for site in blocked_sites:
        filtered_query += f" -site:{site}"

    # 4 검색 실행
    results = []
    try:
        log(" DDGS 인스턴스 생성 중...")
        with DDGS() as ddgs:
            log(" ddgs.text() 호출 중...")
            # 수정 포인트: keywords= 대신 첫 번째 인자로 filtered_query를 전달하거나 
            # 라이브러리 버전에 따라 keywords 대신 q를 사용하기도 합니다. 
            # 가장 안전한 방법은 위치 인자로 첫 번째에 넣는 것입니다.
            raw_results = ddgs.text(
                filtered_query,  # keywords= 대신 직접 전달
                region=region,
                safesearch="moderate",
                timelimit=None, # 최신 버전에서 요구할 수 있는 인자
                max_results=max_results
            )
            
            log(f" raw_results 수신 완료, 파싱 시작")
            if raw_results:
                for r in raw_results:
                    title = r.get("title", "")
                    url = r.get("href", "")
                    body = r.get("body", "")
                    if not title or not url:
                        continue
                    results.append(f"Title: {title}\nURL: {url}\nSnippet: {body}")
            log(f" 파싱 완료: {len(results)}개 결과")
    except Exception as e:
        log(f" 검색 오류: {str(e)}")
        # 에러 발생 시 중단하지 않고 Fallback으로 넘어가기 위해 return 대신 pass 고려 가능

    # 5 중복 제거
    results = list(dict.fromkeys(results))

    # 6 Fallback (결과가 없을 경우)
    if not results:
        log(" 결과 없음, Fallback 검색 시도")
        try:
            with DDGS() as ddgs:
                raw_results = ddgs.text(
                    original_query, # 원본 쿼리로 재시도
                    region="wt-wt",
                    safesearch="moderate",
                    max_results=5
                )
                if raw_results:
                    for r in raw_results:
                        title = r.get("title", "")
                        url = r.get("href", "")
                        body = r.get("body", "")
                        if title and url:
                            results.append(f"Title: {title}\nURL: {url}\nSnippet: {body}")
                log(f" Fallback 결과: {len(results)}개")
        except Exception as e:
            log(f" Fallback 오류: {str(e)}")

    # 7 최종 결과 반환
    if not results:
        log(" 최종 결과 없음")
        return "Search completed but no results were returned."
    
    log(f" 검색 완료: {len(results)}개 반환")
    return "\n\n---\n\n".join(results)

if __name__ == "__main__":
    mcp.run(transport="stdio")