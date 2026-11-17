# -*- coding: utf-8 -*-
"""
eval/eval.py —— 检索质量评测

回答三个问题：
    1. 20 道题里，正确块被召回的比率是多少？（Hit@k）
    2. 正确块平均排第几？（MRR，越接近 1 越好）
    3. top_k=3 和 top_k=5，哪个好？

用法：
    D:\\python3147\\python.exe eval\\eval.py

指标解释（面试要考）：
    Hit@k  = 前 k 个结果里出现正确块的问题数 / 总题数
             「宁可多召回几块，也别漏掉正确的那块」
    MRR    = 每道题算 1/正确块首次出现的名次，再取平均
             例：正确块排第 1 得 1.0，排第 2 得 0.5，排第 3 得 0.33
             「不仅召回了，还要排得靠前」——比 Hit@k 更严格
"""
import json
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # 让 eval 目录里的脚本能 import 根目录的 step2_embed

from step2_embed import embed, VEC_FILE, CHUNK_FILE   # 复用，不重写——检索逻辑只此一份

QUESTIONS = ROOT / "eval" / "eval_questions.json"


def load_index():
    data = np.load(VEC_FILE, allow_pickle=True)
    return data["vectors"], data["idf"], json.loads(CHUNK_FILE.read_text(encoding="utf-8"))


def evaluate(mat, idf, chunks, questions, k):
    """跑一遍全部题目，返回 (hit 数, MRR, 每题明细)"""
    hits, top1,rr_sum, details = 0,0, 0.0, []
    for q in questions:
        v = embed(q["question"], idf)
        scores = mat @ v
        # argsort 从小到大排，[::-1] 反转成从大到小，取前 k 名的块 id
        top = [int(i) for i in np.argsort(scores)[::-1][:k]]
        gold = q["gold_chunks"]
        # 正确块首次出现的名次（没出现记 0）
        first = next((rank for rank, cid in enumerate(top, 1) if cid in gold), 0)
        hit = first > 0
        if first == 1:
            top1 += 1
        hits += hit
        rr_sum += 1 / first if first else 0.0
        details.append((q["id"], q["question"][:18], first, top, gold))
    return hits, top1,  rr_sum / len(questions), details


def main():
    mat, idf, chunks = load_index()
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    n = len(questions)
    print(f"评测集：{n} 题 × 语料 {len(chunks)} 块\n")

    results = {}
    for k in (3, 5):
        hits, top1, mrr, details = evaluate(mat, idf, chunks, questions, k)
        results[k] = (hits, top1, mrr, details)

    # ── 汇总表 ──
    print("=" * 62)
    print(f"{'指标':<10}{'top_k=3':>14}{'top_k=5':>14}")
    print("-" * 62)
    for name, idx in (("Hit@k", 0), ("Hit@1", 1), ("MRR", 2)):
        v3, v5 = results[3][idx], results[5][idx]
        if name in ("Hit@k", "Hit@1"):
            print(f"{name:<10}{f'{v3}/{n} ({v3/n:.0%})':>16}{f'{v5}/{n} ({v5/n:.0%})':>16}")
        else:
            print(f"{name:<10}{v3:>16.3f}{v5:>16.3f}")
    print("=" * 62)

    # ── 明细：只打印有故事的问题 ──
    print("\n明细（名次 0 = 正确块没进前 k；括号里是 [你的标注] 里的正确块）")
    print("-" * 62)
    for qid, qtext, first3, top3, gold in results[3][3]:
        _, _, first5, top5, _ = next(d for d in results[5][3] if d[0] == qid)
        flag = "✅" if first3 else "❌"
        diff = "  ← 5 才救回来" if (not first3 and first5) else ""
        print(f"{flag} id{qid:<3}{qtext:<20} k=3 首位{first3} k=5 首位{first5}  正确块{gold}{diff}")
        if not first5:
            print(f"     实际召回 {top5} vs 你标 {gold}  ← badcase，明晚归因")


if __name__ == "__main__":
    main()
