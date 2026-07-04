"""
Async example: call Anthropic Claude 3.5 Haiku via LiteLLM.
Requires: pip install litellm
Env: ANTHROPIC_API_KEY, GEMINI_API_KEY
"""
import asyncio
from litellm import acompletion, completion_cost


async def main():
    
    response_with_fallback = await acompletion(
        model="anthropic/claude-haiku-4-5-9",
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
        max_tokens=256,
        fallbacks=["gemini/gemini-3.5-flash"],
    )

    text = response_with_fallback.choices[0].message.content
    print(f"Fallback reply ({response_with_fallback.model}):", text)

if __name__ == "__main__":
    asyncio.run(main())
