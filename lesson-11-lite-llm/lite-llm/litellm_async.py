"""
Async example: call Anthropic Claude 3.5 Haiku via LiteLLM.
Requires: pip install litellm
Env: ANTHROPIC_API_KEY, GEMINI_API_KEY
"""

import asyncio
from litellm import acompletion


async def main():
    response = await acompletion(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
        max_tokens=256,
    )
    text = response.choices[0].message.content
    print("Claude reply:", text)


if __name__ == "__main__":
    asyncio.run(main())
