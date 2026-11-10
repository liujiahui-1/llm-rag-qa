import os, requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

resp = requests.post(
    "https://api.deepseek.com/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"model": "deepseek-chat",
          "messages": [{"role": "user", "content": "用一句话解释什么是向量检索"}]},
    timeout=30,
)
print(resp.json()["choices"][0]["message"]["content"])
