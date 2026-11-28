# -*- coding: utf-8 -*-
"""
structured_output_demo.py —— P1 第5项后半段：结构化输出（Structured Output）

上一课 Function Calling：模型"点单"要不要查资料（决定 action）
这一课结构化输出：模型交回来的是一张"填好格子的表"（规定 answer 的格式）

生活比喻：
  Function Calling = 服务员问你要不要加菜，你点单
  结构化输出      = 收银台给你一张发票，金额/日期/项目都填在格子里，机器能直接扫码录入

为什么要这个？
  RAG 的答案要进数据库、要显示在前端，不能是一段散文。
  前端要的是规整的 JSON：{"answer": "...", "sources": [1,3], "confidence": "high"}
  面试时说"我的输出是结构化 JSON，程序能直接解析"，比"模型返回一段文本"专业一个层级。

用法：
    D:\\python3147\\python.exe structured_output_demo.py

跑通后观察三件事：
  1. 返回的 JSON 是不是三个字段都齐
  2. 格式干不干净（能不能被 json.loads 直接解析）
  3. confidence 字段模型是不是真的会按材料的把握程度给
"""
import sys
import json
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from step3_ask import retrieve  # 复用现成的检索函数，一行不重写

API_URL = "https://api.deepseek.com/chat/completions"

# system 里把"表格长什么样"说清楚——这就是结构化输出的提示工程
SYSTEM_PROMPT = (
    "你是学校学生事务问答助手。只依据材料回答，材料里没有就说没有，不要编造。\n"
    "你的回复必须是一个合法的 json 对象，且只包含下面三个字段，不要输出任何多余文字：\n"
    "  answer: 字符串，150字以内的回答，末尾注明信息来自第几页\n"
    "  sources: 数组，元素是你引用材料的块号（整数）\n"
    "  confidence: 字符串，只能是 high / medium / low 三者之一，表示你对答案的把握\n"
    "不要加 markdown 代码块标记（不要写 ```json），直接输出 json 本身。"
)


def build_material(question):
    hits, scores = retrieve(question)
    result = "\n".join(f"块{c['id']}（第{c['page']}页）：{c['text']}" for c in hits)
    return result


def ask_structured(question):
    """发请求 + 强制模型交 JSON"""
    material = build_material(question)
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")

    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                # user 里把检索到的材料和问题一起给模型
                {"role": "user", "content": f"材料：\n{material}\n\n问题：{question}"},
            ],
            "temperature": 0.1,
            "response_format" : {"type" : "json_object"}
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    questions = [
        "转专业需要满足什么条件？",      # 材料里有答案
        "学校食堂几点关门？",            # 材料里大概率没有 —— 看 confidence 会不会降
    ]
    for q in questions:
        print("=" * 62)
        print(f"问题：{q}")
        raw = ask_structured(q)
        print(f"模型原始返回：{raw}")

        # 最后一步：验证能不能直接被程序解析（这就是"结构化"的意义）
        try:
            data = json.loads(raw)
            print(f"解析成功：answer={data['answer']!r}")
            print(f"          sources={data['sources']}  confidence={data['confidence']}")
        except Exception as e:
            print(f"解析失败：{e} —— 说明模型没按格式交，这正是第2道口述题要防的情况")
        print()
