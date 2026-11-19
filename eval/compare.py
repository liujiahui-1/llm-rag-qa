# -*- coding: utf-8 -*-
"""
eval/compare.py —— 两种检索后端的对照评测

同一份语料、同一套 20 道题、同一个指标，只换「怎么把文本变成向量」这一步：
    后端 A：step2_embed 手写 TF-IDF 稀疏向量（认字不认意）
    后端 B：bge-m3 稠密向量（认意）

它回答的是一个简历级问题：
    「换成开源的语义 embedding 模型，检索质量到底提升多少？」
    答案不是感觉，是 Hit@k / Hit@1 / MRR 三列数字的并排对比。

用法：
    D:\\python3147\\python.exe eval\\compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from step2_embed import embed, VEC_FILE, CHUNK_FILE          # noqa: E402  后端 A
from dense_index import build_index, embed_texts            # noqa: E402  后端 B

QUESTIONS = ROOT / "eval" / "eval_questions.json"
BAD_CASES = {2, 4, 6, 16}   # 昨天归因过的 4 道非头名题


def load_all():
    data = np.load(VEC_FILE, allow_pickle=True)
    tfidf_mat, idf = data["vectors"], data["idf"]
    chunks = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    dense_mat, _ = build_index()
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    return tfidf_mat, idf, dense_mat, chunks, questions


def ranks_for(questions, score_fn, n_chunks):
    """对每道题算出「正确块排第几」（全库排序，不在前 n 里就是 n+1）。"""
    rows = []
    for q in questions:
        scores = score_fn(q["question"])
        order = [int(i) for i in np.argsort(scores)[::-1]]
        gold = q["gold_chunks"]
        first = next((r for r, cid in enumerate(order, 1) if cid in gold), n_chunks + 1)
        rows.append((q["id"], q["question"], first, order[:5], gold))
    return rows


def metrics(rows, k):
    """Hit@k / Hit@1 / MRR —— 与 eval.py 里同一套算法。"""
    hit = sum(1 for r in rows if r[2] <= k)
    hit1 = sum(1 for r in rows if r[2] == 1)
    mrr = sum(1 / r[2] if r[2] <= k else 0.0 for r in rows) / len(rows)
    return hit, hit1, mrr


def main():
    tfidf_mat, idf, dense_mat, chunks, questions = load_all()
    n = len(questions)

    tfidf_rows = ranks_for(questions, lambda q: tfidf_mat @ embed(q, idf), len(chunks))
    dense_rows = ranks_for(questions, lambda q: dense_mat @ embed_texts([q])[0], len(chunks))

    print("\n" + "=" * 62)
    print("检索后端对照评测   语料 15 块 · 评测集 20 题")
    print("=" * 62)
    print(f"{'指标':<10}{'TF-IDF(稀疏)':>16}{'bge-m3(稠密)':>16}{'变化':>14}")
    for k in (3, 5):
        a_hit, _, a_mrr = metrics(tfidf_rows, k)
        b_hit, _, b_mrr = metrics(dense_rows, k)
        print(f"{'Hit@' + str(k):<10}{f'{a_hit}/{n} ({a_hit/n:.0%})':>16}"
              f"{f'{b_hit}/{n} ({b_hit/n:.0%})':>16}{f'{(b_hit-a_hit)/n:+.0%}':>14}")
    # metrics() 返回 (hit, hit1, mrr) 三个值，Hit@1 要取第二个
    _, a_hit1, a_mrr = metrics(tfidf_rows, len(chunks))
    _, b_hit1, b_mrr = metrics(dense_rows, len(chunks))
    print(f"{'Hit@1':<10}{f'{a_hit1}/{n} ({a_hit1/n:.0%})':>16}"
          f"{f'{b_hit1}/{n} ({b_hit1/n:.0%})':>16}{f'{(b_hit1-a_hit1)/n:+.0%}':>14}")
    print(f"{'MRR':<10}{a_mrr:>16.3f}{b_mrr:>16.3f}{b_mrr-a_mrr:>+14.3f}")

    print("\n" + "-" * 62)
    print("逐题名次对比（只列名次有变化或不是第 1 的题）")
    print("-" * 62)
    print(f"{'id':<4}{'题':<22}{'TF-IDF':>8}{'bge-m3':>8}{'':>6}")
    for a, b in zip(tfidf_rows, dense_rows):
        if a[2] == b[2] == 1:
            continue
        flag = "✅ 治好了" if b[2] == 1 and a[2] > 1 else ("⚠️ 变差" if b[2] > a[2] else "")
        mark = " *" if a[0] in BAD_CASES else "  "
        print(f"{a[0]:<4}{a[1][:20]:<22}{a[2]:>8}{b[2]:>8}   {flag}{mark}")

    # 词义相似度：为什么稠密能治好「同义换字」
    print("\n" + "-" * 62)
    print("同一对句子的相似度（TF-IDF 只认字面，bge-m3 认意思）")
    print("-" * 62)
    pairs = [
        ("转专业有什么限制", "下列情形不得转专业", "同义：限制 vs 不得"),
        ("助学贷款最多能贷多少", "生源地信用助学贷款，每人每年最高 16000 元", "同义：贷款 vs 信用助学贷款"),
        ("新生入学要带什么", "新生报到须携带录取通知书", "同义：带什么 vs 携带"),
    ]
    print(f"{'句对':<34}{'TF-IDF':>10}{'bge-m3':>10}")
    for x, y, note in pairs:
        a = float(embed(x, idf) @ embed(y, idf))
        b = float(embed_texts([x])[0] @ embed_texts([y])[0])
        print(f"{note:<34}{a:>10.3f}{b:>10.3f}")


if __name__ == "__main__":
    main()
