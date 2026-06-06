from ollama import chat

response = chat(
    model='joker',
    messages=[{'role': 'user', 'content': 'Hello!'}],
)

print(response.message.content)
