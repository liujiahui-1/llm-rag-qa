# -*- coding: utf-8 -*-
"""
validate_demo.py —— 结构化输出第二课：程序当质检员

背景（昨晚的发现）：
  模型一边回答"材料里没有食堂关门的信息"，一边 confidence 给了 high。
  模型自报的置信度不可全信 → 模型输出之后，必须有一道程序质检。

质检管两件事：
  1. 字段质检：三个字段（answer/sources/confidence）缺一不可
  2. 置信度校验：answer 明明说"没有答案"，confidence 却是 high → 强制降级

用法：
    D:\\python3147\\python.exe validate_demo.py

复用昨晚的成果：直接 import 昨天写的 ask_structured，一行不重写。
"""
import json

from structured_output_demo import ask_structured  # 复用昨晚整个链路

REQUIRED = ["answer", "sources", "confidence"]  # 质检清单：必须有这三个字段

# 这些词一出现，说明模型的回答是"材料里没有答案"
NO_ANSWER_KEYWORDS = ["没有提到", "没有关于", "材料中没有", "未提到", "没有相关"]


def validate(data):
    problems = []  # 问题清单，质检发现一条记一条

    for k in REQUIRED:
        if k not in data:
            problems.append(f"缺字段：{k}")

    no_answer = False
    for k in NO_ANSWER_KEYWORDS:
        if k in data["answer"]:
            no_answer = True
    if no_answer == True and data["confidence"] == "high":
        problems.append("回答说不存在，置信度却是 high，已强制降级")
        data["confidence"] = "low"

    return len(problems) == 0, problems


if __name__ == "__main__":
    questions = [
        "转专业需要满足什么条件？",   # 有答案 → 应该通过质检
        "学校食堂几点关门？",         # 没答案 → 昨晚它 confidence 给了 high，看今天质检抓不抓得住
    ]
    for q in questions:
        print("=" * 62)
        print(f"问题：{q}")
        raw = ask_structured(q)
        data = json.loads(raw)
        print(f"模型交的表：answer={data.get('answer', '（缺失）')[:40]}...  "
              f"sources={data.get('sources', '（缺失）')}  confidence={data.get('confidence', '（缺失）')}")

        ok, problems = validate(data)
        if ok:
            print(f"质检：通过 ✓（0 个问题）")
        else:
            for p in problems:
                print(f"质检：发现问题 → {p}")
            print(f"降级后的 confidence：{data['confidence']}")
        print()
