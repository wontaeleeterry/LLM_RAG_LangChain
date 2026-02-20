from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Optional
import yfinance as yf
from datetime import datetime, timedelta

class StockRequest(BaseModel):
    symbol: str = Field(description="종목 코드 (예: 005930, AAPL)")
    date: Optional[str] = Field(default=None, description="조회할 날짜 (YYYY-MM-DD 형식, 미입력 시 최신 종가)")

mcp = FastMCP("stock_mcp")

@mcp.tool()
def get_stock_price(input: StockRequest) -> str:
    """
    yfinance를 사용하여 한국/해외 주식의 가격을 조회합니다.
    특정 날짜(YYYY-MM-DD)를 입력하면 해당 날짜의 종가를 반환합니다.
    """
    code = input.symbol.strip().upper()
    target_date = input.date
    
    # 1. 티커 후보군 생성 (재시도 전략)
    if code.endswith(".KS") or code.endswith(".KQ"):
        candidate_tickers = [code]
    elif code.isdigit() and len(code) == 6:
        candidate_tickers = [f"{code}.KS", f"{code}.KQ"]
    else:
        candidate_tickers = [code]

    last_exception = None

    for ticker in candidate_tickers:
        try:
            stock = yf.Ticker(ticker)
            
            if target_date:
                # 특정 날짜 조회 로직
                # yfinance history는 end 날짜를 포함하지 않으므로 +1일을 해줍니다.
                start_dt = datetime.strptime(target_date, "%Y-%m-%d")
                end_dt = start_dt + timedelta(days=1)
                
                df = stock.history(start=start_dt.strftime("%Y-%m-%d"), 
                                   end=end_dt.strftime("%Y-%m-%d"))
                
                if not df.empty:
                    price = df["Close"].iloc[0]
                    date_str = df.index[0].strftime("%Y-%m-%d")
                    return f"[{date_str}] {ticker}의 종가는 {price:,.2f}원(또는 해당 통화)입니다."
                else:
                    # 해당 날짜에 데이터가 없는 경우 (주말, 휴장일 등) 다음 티커 시도 전 체크
                    continue
            else:
                # 최신 가격 조회 로직 (기존 로직 유지)
                df = stock.history(period="1d")
                if not df.empty:
                    price = df["Close"].iloc[-1]
                    date_str = df.index[-1].strftime("%Y-%m-%d")
                    return f"[최신: {date_str}] {ticker}의 가격은 {price:,.2f}원입니다."
                continue

        except Exception as e:
            last_exception = e
            continue

    msg = f"'{code}'에 대한 정보를 찾을 수 없습니다."
    if target_date:
        msg += f" (날짜: {target_date} - 주말이나 공휴일인지 확인해주세요.)"
    return f"오류: {msg} (상세: {last_exception})"

if __name__ == "__main__":
    mcp.run()