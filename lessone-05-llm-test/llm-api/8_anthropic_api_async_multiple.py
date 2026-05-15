import os
import asyncio
from anthropic import AsyncAnthropic, DefaultAioHttpClient


async def hello_claude(client: AsyncAnthropic) -> None:
    print("Before call")
    message = await client.messages.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "Hello, Claude",
                }
            ],
            model="claude-haiku-4-5",
        )
    print(message.content)
    print("After call")


async def main() -> None:
    client = AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        http_client=DefaultAioHttpClient(),
    )

    tasks = [hello_claude(client) for _ in range(5)]
    await asyncio.gather(*tasks)
    print("All calls completed")


if __name__ == "__main__":
    asyncio.run(main())