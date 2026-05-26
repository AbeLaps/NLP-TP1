from groq import Groq
from openai import OpenAI
import os

prompt = "What is the capital of France?"


client = OpenAI(

    api_key=os.environ.get("GROQ_API_KEY"),

    base_url="https://api.groq.com/openai/v1",

)


completion = client.chat.completions.create(
    model="qwen/qwen3-32b",
    messages=[
      {
        "role": "user",
        "content": prompt
      }
    ],
    temperature=0.6,
    max_completion_tokens=4096,
    top_p=0.95,
    reasoning_effort="default",
    stream=True,
    stop=None
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
    print("\n")

print("\n\nFull response:")
