from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
import yfinance as yf

class StockRequest(BaseModel):
    symbol: str  # 종목 코드

mcp = FastMCP("stock_mcp")

@mcp.tool()
def get_korean_stock_price(input: StockRequest) -> str:
    """
    yfinance를 사용하여 한국 주식의 가격을 조회합니다.
    최대한 안정적이고 빠른 방법을 사용합니다.
    """
    code = input.symbol.strip()

    # 접미사 자동 판별
    if code.endswith(".KQ") or code.endswith(".KS"):
        ticker = code
    else:
        if code.startswith(("0", "1", "2")):
            ticker = code + ".KQ"
        else:
            ticker = code + ".KS"

    try:
        stock = yf.Ticker(ticker)

        # 우선 fast_info를 먼저 사용
        price = stock.fast_info.get("last_price") or stock.fast_info.get("regular_market_price")
        name = stock.fast_info.get("shortName") or ticker

        # fast_info 실패 시 history 사용 (최근 1분봉 데이터)
        if price is None:
            df = stock.history(period="1d", interval="1m")
            if not df.empty:
                price = df["Close"].iloc[-1]
            else:
                raise ValueError("가격 정보를 가져올 수 없습니다.")

        return f"{name}({ticker})의 현재 가격은 {price:.2f}원입니다."

    except Exception as e:
        return f"오류: {e}"

if __name__ == "__main__":
    mcp.run()
