# -*- coding: utf-8 -*-
"""eval/hybrid.py —— 混合检索：用 RRF 融合「稀疏」与「稠密」两路名次

为什么需要它：
    TF-IDF 靠字面命中，原文一字不差就赢；bge-m3 靠语义，换个说法也认得。
    两边都有对方学不会的本事，也都有各自的盲区（见 model_compare.md）。
    所以不是二选一，而是「两路都召回，再把名次合并」。

RRF（Reciprocal Rank Fusion，倒数排名融合）怎么合并：
    不看两边的原始分数——稀疏是余弦相似度、稠密也是余弦相似度，但量纲和分布不同，
    直接相加没有意义。只按「名次」算分：

        score(块) = 1/(k + 第A路名次) + 1/(k + 第B路名次)      k=60

    名次越靠前，得分越高；两路都靠前的块自然胜出。
    k 的作用是压平差距：k 大时第 1 名和第 2 名得分很接近，融合更看重「两路是否都同意」；
    k 小时更偏向「某一路的绝对第一」。工业界默认 60。

用法：
    D:\\python3147\\python.exe eval\\hybrid.py
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
# 25 题里两路各自失守首位的题（3/10/13/21/25 是稠密丢的，2/4/6/16/22/24 是稀疏丢的）
SPARSE_LOST = {2, 4, 6, 16, 22, 24}
DENSE_LOST = {3, 10, 13, 21, 25}


def load_all():
    data = np.load(VEC_FILE, allow_pickle=True)
    tfidf_mat, idf = data["vectors"], data["idf"]
    chunks = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    dense_mat, _ = build_index()
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    return tfidf_mat, idf, dense_mat, chunks, questions


def orders_for(questions, score_fn, n_chunks):
    """对每道题返回「全库块的完整名次列表」（从最相关到最不相关）。"""
    out = []
    for q in questions:
        scores = score_fn(q["question"])
        out.append([int(i) for i in np.argsort(scores)[::-1]])
    return out


def rank_of(order, gold):
    """正确块在这份名次里排第几；一个都没进前 n 就是 n+1。"""
    return next((r for r, cid in enumerate(order, 1) if cid in gold), len(order) + 1)


def rrf_fuse(order_a, order_b, k=60):
    """把两路检索的名次融合成一个新的名次列表。

    参数:
        order_a: 后端 A（TF-IDF）的块 id 列表，按相关度从高到低
        order_b: 后端 B（bge-m3）的块 id 列表，同上
        k:       平滑常数，默认 60

    返回:
        fused: 融合后的块 id 列表，长度与输入相同，按融合得分从高到低排列

    公式（每个块算一份得分，两路相加，rank 从 1 开始）:
        score(块) = 1/(k + rank_a) + 1/(k + rank_b)

    提示:
        - 用一个 dict 存「块 id -> 得分」，两路各遍历一遍，同名块得分累加
        - 排序时得分高的在前；得分相同时随意（这里不会出现并列影响结论的情况）
    """
    score = {}
    for r, cid in enumerate(order_a, 1):
        score[cid] = score.get(cid, 0.0) + 1 / (k + r)
    for r, cid in enumerate(order_b, 1):
        score[cid] = score.get(cid, 0.0) + 1 / (k + r)
    return sorted(score, key=score.get, reverse=True)


def metrics(orders, questions, k):
    """Hit@k / Hit@1 / MRR。"""
    ranks = [rank_of(o, q["gold_chunks"]) for o, q in zip(orders, questions)]
    hit = sum(1 for r in ranks if r <= k)
    hit1 = sum(1 for r in ranks if r == 1)
    mrr = sum(1 / r if r <= k else 0.0 for r in ranks) / len(ranks)
    return hit, hit1, mrr


def main():
    tfidf_mat, idf, dense_mat, chunks, questions = load_all()
    n = len(questions)
    n_chunks = len(chunks)

    a_orders = orders_for(questions, lambda q: tfidf_mat @ embed(q, idf), n_chunks)
    b_orders = orders_for(questions, lambda q: dense_mat @ embed_texts([q])[0], n_chunks)
    h_orders = [rrf_fuse(a, b) for a, b in zip(a_orders, b_orders)]

    print("\n" + "=" * 70)
    print(f"混合检索（RRF k=60）   语料 {n_chunks} 块 · 评测集 {n} 题")
    print("=" * 70)
    print(f"{'指标':<10}{'TF-IDF':>14}{'bge-m3':>14}{'hybrid':>14}{'vs 最好单路':>16}")
    for name, k in (("Hit@3", 3), ("Hit@1", n_chunks)):
        a = metrics(a_orders, questions, k)
        b = metrics(b_orders, questions, k)
        h = metrics(h_orders, questions, k)
        idx = 1 if name == "Hit@1" else 0
        best = max(a[idx], b[idx])
        print(f"{name:<10}{f'{a[idx]}/{n} ({a[idx]/n:.0%})':>14}"
              f"{f'{b[idx]}/{n} ({b[idx]/n:.0%})':>14}"
              f"{f'{h[idx]}/{n} ({h[idx]/n:.0%})':>14}"
              f"{f'{h[idx]-best:+d} 题':>16}")
    a_mrr, b_mrr, h_mrr = metrics(a_orders, questions, n_chunks)[2], \
        metrics(b_orders, questions, n_chunks)[2], metrics(h_orders, questions, n_chunks)[2]
    print(f"{'MRR':<10}{a_mrr:>14.3f}{b_mrr:>14.3f}{h_mrr:>14.3f}"
          f"{f'{h_mrr-max(a_mrr, b_mrr):+.3f}':>16}")

    print("\n" + "-" * 70)
    print("逐题名次（只看 hybrid 没拿第一，或两路曾经失守的题）")
    print("-" * 70)
    print(f"{'id':<4}{'题':<24}{'TF':>5}{'稠密':>6}{'混合':>6}   说明")
    for a, b, h, q in zip(a_orders, b_orders, h_orders, questions):
        ra, rb, rh = rank_of(a, q["gold_chunks"]), rank_of(b, q["gold_chunks"]), rank_of(h, q["gold_chunks"])
        if rh == 1 and ra == 1 and rb == 1:
            continue
        qid = q["id"]
        tags = []
        if qid in SPARSE_LOST:
            tags.append("稀疏失守")
        if qid in DENSE_LOST:
            tags.append("稠密失守")
        if rh == 1:
            tags.append("✅ 混合捞回")
        elif rh > min(ra, rb):
            tags.append("⚠️ 混合更差")
        print(f"{qid:<4}{q['question'][:22]:<24}{ra:>5}{rb:>6}{rh:>6}   {'/'.join(tags)}")

    print("\n" + "-" * 70)
    print("今天的目标：hybrid 的 Hit@1 和 MRR 都要高于 76% / 0.873 和 80% / 0.900 中的较好者")


if __name__ == "__main__":
    main()
