# -*- coding: utf-8 -*-
"""
step3_ask.py —— RAG 最后一环：检索到的材料 + 问题 -> 拼 Prompt -> 大模型回答

用法：
    python step3_ask.py "补考成绩怎么记载"

到这一步为止，完整链路 = step1 切分 + step2 检索 + 这里的拼装与生成。
"""
import sys
import os
import json
import numpy as np
import requests
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))  # 让 python 能 import 同目录的 step2_embed

# 复用 step2 的全部成果：柜子、向量、检索，一行都不用重写
from step2_embed import embed, CHUNK_FILE, VEC_FILE

TOP_K = 3  # 每次给模型几块材料：少了可能漏答案，多了浪费字数还可能互相干扰


def retrieve(question, k=TOP_K):
    """问题变向量 -> 和 15 块比相似度 -> 取最像的 k 块（step2 已经做过的一切）"""
    data = np.load(VEC_FILE, allow_pickle=True)
    mat, idf = data["vectors"], data["idf"]
    chunks = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    q = embed(question, idf)
    scores = mat @ q
    top = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top], [float(scores[i]) for i in top]


def build_prompt(question, hits):
    """把材料包进模板里发出去。所谓 Prompt 工程，最基本的形式就是拼模板。
    三条规则：
      1. 告诉模型「只依据材料答」——防止它用训练时见过的其他学校的规定来编
      2. 材料带上页码——模型才能在答案里标注出处
      3. 限定长度——回答啰嗦会盖住关键信息"""
    parts = []
    for rank, (c, s) in enumerate(hits, 1):
        parts.append(f"[材料{rank}]（第{c['page']}页）\n{c['text']}")
    material = "\n\n".join(parts)
    return (
        "请只依据下面的材料回答问题。如果材料里没有答案，就回答"
        "「材料中没有提到」，不要编造。\n\n"
        f"{material}\n\n"
        f"问题：{question}\n\n"
        "回答要求：不超过 150 字；说明信息来自第几页。"
    )


def ask_deepseek(prompt):
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "deepseek-chat",
            "messages": [
                # system 定角色：一份"岗位说明书"，模型全程遵守
                {"role": "system", "content": "你是学校学生事务问答助手，回答必须严谨、简短、有出处。"},
                {"role": "user", "content": prompt},
            ],
            # 温度：0=每次都选概率最高的词，输出稳定；1=更随机更有创意
            # 事实问答必须调低——同一问题问三遍得到三个答案就是事故
            "temperature": 0.1,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "补考成绩怎么记载"

    print(f"问题：{question}")
    print()
    hits, scores = retrieve(question)

    print("=" * 62)
    print("第 1 步：检索（step2 的活）—— 挑出最像的 3 块")
    print("=" * 62)
    for c, s in zip(hits, scores):
        print(f"  相似度 {s:.3f} | 第 {c['page']} 页 | 块 {c['id']}")

    prompt = build_prompt(question, list(zip(hits, scores)))
    print()
    print("=" * 62)
    print("第 2 步：拼 Prompt —— 下面这段话就是发给模型的全部内容")
    print("=" * 62)
    print(prompt[:500])
    print("  ...（中间是三块材料原文，省略）")

    print()
    print("=" * 62)
    print("第 3 步：模型回答")
    print("=" * 62)
    print(ask_deepseek(prompt))
