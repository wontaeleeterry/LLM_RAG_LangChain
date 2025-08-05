from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
import yfinance as yf

class StockRequest(BaseModel):
    symbol: str  # 종목 코드

mcp = FastMCP("stock_mcp")

@mcp.tool()
def get_korean_stock_price(input: StockRequest) -> str:
    """
    한국 주식의 실시간 가격을 조회합니다.
    """
    code = input.symbol.strip()

    if code.endswith(".KQ") or code.endswith(".KS"):
        ticker = code
    else:
        if code.startswith("0") or code.startswith("1") or code.startswith("2"):
            ticker = code + ".KQ"  # 코스닥
        else:
            ticker = code + ".KS"  # 코스피

    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get("regularMarketPrice")
        name = stock.info.get("shortName", ticker)
        if price is None:
            raise ValueError("가격 정보를 찾을 수 없습니다.")
        return f"{name}({ticker})의 현재 가격은 {price}원입니다."
    except Exception as e:
        return f"오류: {e}"

if __name__ == "__main__":
    mcp.run()
