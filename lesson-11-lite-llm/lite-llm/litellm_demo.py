"""
Simple example: call Anthropic Claude, OpenAI GPT-5.4 mini, and Gemini via LiteLLM.
Requires: pip install litellm
Env: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
"""
from dotenv import load_dotenv
from litellm import completion

# Load ANTHROPIC_API_KEY (and any other keys) from a .env file in this folder.
load_dotenv()

# NOTE: With LiteLLM you do NOT need an anthropic.Anthropic() client.
# completion() reads ANTHROPIC_API_KEY from the environment automatically.

def main():
    response = completion(
        # model="gemini/gemini-3.5-flash",
        model="anthropic/claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
        max_tokens=256,
        num_retries=3,
    )
    text = response.choices[0].message.content
    print("Claude reply:", text)

if __name__ == "__main__":
    main()
