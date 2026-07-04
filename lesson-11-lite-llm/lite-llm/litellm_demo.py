"""
Simple example: call Anthropic Claude, OpenAI GPT-5.4 mini, and Gemini via LiteLLM.
Requires: pip install litellm
Env: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
"""
import os
from litellm import completion

# Use your API keys (or set them in the environment)
# os.environ["ANTHROPIC_API_KEY"] = "your-key"
# os.environ["OPENAI_API_KEY"] = "your-key"

def main():
    response = completion(
        # model="gemini/gemini-3.5-flash",
        model="anthropic/claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
        max_tokens=256,
        number_or_retries=3,
    )
    text = response.choices[0].message.content
    print("Claude reply:", text)

if __name__ == "__main__":
    main()
