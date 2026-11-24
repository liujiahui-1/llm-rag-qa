# -*- coding: utf-8 -*-
"""
fc_demo.py —— Function Calling 第一课：让模型自己决定"要不要查资料"

之前的 RAG 流程是死板的：不管什么问题，都先检索 3 块材料再问模型。
今天改成"智能版"：把检索注册成工具，模型自己判断——
  需要资料的问题（如"转专业条件"）→ 它主动点单要检索
  不需要的问题（如"1+1等于几"）→ 它直接回答，不浪费一次检索

用法：
    D:\\python3147\\python.exe fc_demo.py

运行后观察两件事：
  1. 哪个问题触发了工具调用，哪个没有
  2. 最终回答的内容质量有什么区别
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

# ============================================================
# TODO 1（你来写）：工具说明书
#
# tools 是一份 JSON，告诉模型"你有一个叫 search_knowledge 的工具，
# 它接收一个参数 query（要检索的问题）"。
#
# 结构提示（照 OpenAI 函数调用格式，DeepSeek 完全兼容）：
#   TOOLS = [{
#       "type": "function",
#       "function": {
#           "name": "工具名",
#           "description": "给模型看的介绍：什么时候该用这个工具",
#           "parameters": {
#               "type": "object",
#               "properties": {
#                   "参数名": {"type": "string", "description": "参数说明"}
#               },
#               "required": ["参数名"]
#           }
#       }
#   }]
#
# description 写得越清楚，模型判断得越准——这是"提示工程"的一部分。
# ============================================================
TOOLS = [{
    "type": "function",
    "function" : {
        "name" : "search_knowledge",
        "description" : "检索学校学生事务知识库。凡是涉及学校规定、政策、流程的问题（如转专业、免修、退学、成绩记载），必须先调用本工具查询原文材料，再依据材料回答。",
        "parameters" : {
            "type" : "object",
            "properties" : {
                "query" : {
                    "type" : "string", "description": "要检索的问题或关键词，用中文"
                }
            },
            "required" : ["query"]
        }
    }
}]

SYSTEM_PROMPT = (
    "你是学校学生事务问答助手。涉及学校规定、政策、流程的问题，"
    "请先调用 search_knowledge 工具查询资料，只依据查到的材料回答；"
    "常识性或闲聊类问题直接回答，不必调用工具。"
)


def chat(messages):
    """发一轮对话给 DeepSeek。messages 里的内容每轮都在变长。"""
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "deepseek-chat",
            "messages": messages,
            "tools": TOOLS,
            "temperature": 0.1,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


def run_agent(question):
    """核心循环：问 → 模型要么点单（tool_calls）要么直接给答案。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for turn in range(1, 4):  # 最多 3 轮，防止模型无限点单
        msg = chat(messages)
        calls = msg.get("tool_calls")

        if not calls:
            # 模型没点单 → 它认为已经有足够信息，直接输出答案
            print(f"  [第 {turn} 轮] 模型直接回答（没有调用工具）")
            return msg["content"]

        print(f"  [第 {turn} 轮] 模型点单了 {len(calls)} 个工具：")
        messages.append(msg)  # 必须把模型的"点单"记回对话，它才知道自己点过

        for tc in calls:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            print(f"    工具 = {name}，参数 = {args}")

            if name == "search_knowledge":
                # ============================================================
                # TODO 2（你来写）：替模型干这单活
                #
                # 三步：
                #   1. 调 retrieve(args["query"]) 拿到 hits（ remember step3 的返回是 (hits, scores)）
                #   2. 把每块材料拼成一段可读文字（带页码），类似：
                #      "第X页：原文..."
                #   3. 结果存进变量 result（字符串类型）
                # ============================================================
                hits, scores = retrieve(args["query"])
                result = "\n".join(f"第{c['page']}页：{c['text']}" for c in hits)
            else:
                result = f"未知工具：{name}"

            # 把干活结果念回给模型——role 必须是 "tool"，id 必须对上它点的单
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    return "（超过最大轮数，强制停止）"


if __name__ == "__main__":
    questions = [
        "转专业需要满足什么条件？",   # 应该触发检索
        "你好，用一句话介绍你自己",    # 不应该触发检索
    ]
    for q in questions:
        print("=" * 62)
        print(f"问题：{q}")
        answer = run_agent(q)
        print()
        print(f"最终回答：{answer}")
        print()
