# -*- coding: utf-8 -*-
"""
step3_解剖.py —— 把 step3_ask.py 拆开看，一步一打印

四段演示：
  1. 拼 Prompt：一块一块拼给你看（像填请假条模板）
  2. 发出去的请求长什么样（像填寄件单）
  3. 收回来的 JSON 怎么一层层拆出答案（像拆快递套娃）
  4. 温度实验：同一个问题，低温 vs 高温，看模型稳不稳
"""
import os
import json
import numpy as np
import requests
from dotenv import load_dotenv
from step2_embed import embed, CHUNK_FILE, VEC_FILE

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
Q = "宿舍最晚几点关门"


def line(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ================= 演示 1：拼 Prompt =================
line("演示 1：拼 Prompt —— 三块积木，从上往下拼")

data = np.load(VEC_FILE, allow_pickle=True)
mat, idf = data["vectors"], data["idf"]
chunks = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
q = embed(Q, idf)
scores = mat @ q
top = np.argsort(scores)[::-1][:3]

print(f"问题：{Q}")
print("\n检索（step2 的活，你已经学过，一行带过）——捞出 3 块：")
for rank, i in enumerate(top, 1):
    c = chunks[i]
    print(f"   材料{rank}：第 {c['page']} 页，开头「{c['text'][:18]}...」")

print("\n现在开始拼。所谓 Prompt 工程，最基本的形式就是：")
print("   模板 = 规则句 + 材料区 + 问题句")

rules = "请只依据下面的材料回答问题。材料里没有就回答「材料中没有提到」，不要编造。"
print("\n【第 1 块：规则句】")
print("   " + rules)
print("   ↑ 这句是防编造的锁。没有它，模型会像刚才那样自己编 500 字。")

material_parts = []
for rank, i in enumerate(top, 1):
    c = chunks[i]
    material_parts.append(f"[材料{rank}]（第{c['page']}页）\n{c['text']}")
material = "\n\n".join(material_parts)

print("\n【第 2 块：材料区】")
print(f"   三块文字用空行连起来，共 {len(material)} 个字")
print(f"   为什么带页码？——模型才能在答案里说「出自第几页」，这就是溯源")

question_part = f"问题：{Q}\n\n回答要求：不超过 150 字；说明信息来自第几页。"
print("\n【第 3 块：问题句】")
print("   " + question_part.replace("\n", "  "))

prompt = rules + "\n\n" + material + "\n\n" + question_part
print(f"\n>>> 拼好了！整段 Prompt 共 {len(prompt)} 个字。")
print(">>> 代码里 build_prompt() 干的全部事情，就是这三步加号。")

# ================= 演示 2：发出去的请求 =================
line("演示 2：发给 DeepSeek 的请求 —— 像填一张寄件单")

body = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "你是学校学生事务问答助手，回答必须严谨、简短、有出处。"},
        {"role": "user", "content": prompt},
    ],
    "temperature": 0.1,
}
print("请求 = 地址 + 门禁卡 + 包裹")
print()
print("   地址：   https://api.deepseek.com/chat/completions")
print("   门禁卡： headers 里的 Bearer sk-****        ← 证明你充过钱")
print("   包裹(json)：")
print("      model        = deepseek-chat      ← 用哪个模型")
print("      temperature  = 0.1                ← 随机旋钮（演示 4 细说）")
print("      messages     = 两条消息的列表：")
print("         [0] role=system   ← 岗位说明书：你是谁、怎么答（全程生效）")
print("         [1] role=user     ← 这一单的活：拼好的 800 字 Prompt")

# ================= 演示 3：拆返回的 JSON =================
line("演示 3：收回来的 JSON —— 像拆快递套娃，一层一层开")

print("发出请求，等模型写完（这一步要几秒钟）...\n")
resp = requests.post(
    "https://api.deepseek.com/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json=body,
    timeout=60,
)
full = resp.json()

print("① 先看包裹最外层有哪些盒子：", list(full.keys()))
print("\n② 打开 choices 盒子——是个列表（本次只让它答一条，所以只有 1 个元素）")
print("   full['choices'] 长度 =", len(full["choices"]))
print("\n③ 取第 0 个盒子看看里面：")
inner = full["choices"][0]
for k, v in inner.items():
    shown = (str(v)[:50] + "...") if len(str(v)) > 50 else v
    print(f"   {k}: {shown}")
print("\n④ message 里还有一层：")
msg = inner["message"]
print("   role    =", msg["role"], "  ← assistant = 模型的回答")
print("   content =", repr(msg["content"][:60]) + "...")

answer = full["choices"][0]["message"]["content"]
print("\n⑤ 最终答案（走完四层：json → choices → [0] → message → content）：")
print("   " + answer)

# ================= 演示 4：温度实验 =================
line("演示 4：温度旋钮 —— 同一个问题问三遍，看模型稳不稳")


def ask(temp):
    body2 = dict(body)
    body2["temperature"] = temp
    r = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=body2,
        timeout=60,
    )
    return r.json()["choices"][0]["message"]["content"]


print("temperature = 0.1（低温，几乎不随机）：")
print("   " + answer)
print("\ntemperature = 1.5（高温，很随机），连问两遍：")
for i in range(2):
    print(f"   第{i+1}遍：" + ask(1.5))

print("""
结论：
   低温 = 问三遍答案基本一样 → 事实问答必须用它
   高温 = 每遍说法都不一样   → 写文案、头脑风暴才用它
   你的系统查成绩答错一次都不行，所以 step3_ask.py 里写死 0.1
""")
