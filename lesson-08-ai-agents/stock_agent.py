import yfinance as yf
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from llm_factory import build_llm

load_dotenv()


SYSTEM_PROMPT = """
You are a helpful stock assistant.

Use get_stock_info when the user asks about:
- current stock price
- quote
- current market data
- day high, day low, volume, or currency

Use get_stock_year_performance when the user asks about:
- whether a stock went up or down over the past year
- one-year performance
- 12-month performance
- how a stock performed during the last year

Do not invent live market data or historical performance.
If the user does not provide a ticker symbol, ask for one.
Summarize tool results clearly for the user.
Do not provide financial advice or tell the user to buy or sell a stock.
"""


@tool
def get_stock_info(symbol: str) -> str:
    """
    Get basic stock market data for a ticker symbol.

    Args:
        symbol: Stock ticker symbol, for example MSFT, AAPL, TSLA, or NVDA.
    """
    symbol = symbol.strip().upper()

    if not symbol:
        return "Error: Please provide a stock ticker symbol."

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info:
            return f"Error: No market data found for {symbol}."

        name = info.get("longName") or info.get("shortName") or symbol

        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        currency = info.get("currency", "")
        day_high = info.get("dayHigh")
        day_low = info.get("dayLow")
        volume = info.get("volume")

        if price is None:
            return f"Error: Could not find a current price for {symbol}."

        lines = [
            f"Ticker: {symbol}",
            f"Company: {name}",
            f"Current price: {price} {currency}".strip(),
        ]

        if day_high is not None:
            lines.append(f"Day high: {day_high}")

        if day_low is not None:
            lines.append(f"Day low: {day_low}")

        if volume is not None:
            lines.append(f"Volume: {volume}")

        return "\n".join(lines)

    except Exception:
        return f"Error: Could not fetch stock data for {symbol}."

@tool
def get_stock_year_performance(symbol: str) -> str:
    """Check whether a stock went up or down over the past year."""
    symbol = symbol.strip().upper()

    if not symbol:
        return "Error: Please provide a stock ticker symbol."

    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="1y")

        if history.empty:
            return f"Error: No historical price data found for {symbol}."

        start_price = history["Close"].iloc[0]
        end_price = history["Close"].iloc[-1]

        if start_price == 0:
            return f"Error: Invalid start price found for {symbol}."

        change = end_price - start_price
        change_percent = (change / start_price) * 100

        if change > 0:
            direction = "up"
        elif change < 0:
            direction = "down"
        else:
            direction = "flat"

        return (
            f"{symbol} went {direction} over the past year.\n"
            f"Start price: {start_price:.2f}\n"
            f"End price: {end_price:.2f}\n"
            f"Change: {change_percent:.2f}%"
        )

    except Exception:
        return f"Error: Could not fetch historical data for {symbol}."

def build_agent():
    model = build_llm()

    return create_agent(
        model=model,
        tools=[get_stock_info, 
               get_stock_year_performance
        ],
        system_prompt=SYSTEM_PROMPT,
    )


def query_agent(agent, user_input: str) -> str:
    """
    Send a user question to the agent and return the final answer.
    """
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        }
    )

    last_message = result["messages"][-1]

    if isinstance(last_message, AIMessage):
        return last_message.content

    return str(last_message)


def main():
    print("Loading Stock Agent...")

    agent = build_agent()

    print("Stock Agent is ready.")
    print("Ask about stock prices, quotes, or market data.")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        if not user_input:
            print("Please enter a question.")
            continue

        try:
            answer = query_agent(agent, user_input)
            print(f"\nAssistant: {answer}")

        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()