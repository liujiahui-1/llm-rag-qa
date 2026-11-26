# -*- coding: utf-8 -*-
"""eval/topn_sweep.py —— TOP_N 取舍实验：精排到底该取前几名

为什么做这个实验：
    rerank.py 里 TOP_N = 6 是照业界经验拍的，没验证过。
    面试官一定会追问一句「为什么是 6？取 3 会怎样？取满 15 呢？」
    这个脚本把 3 / 6 / 10 / 15 四档一次跑完，用数字回答，不靠感觉。

四个档位的含义（语料一共 15 块）：
     3  → 只精排前 3 名，覆盖全库 20%
     6  → 现在用的，覆盖 40%
    10  → 覆盖 67%
    15  → 全库都精排（等于拿掉"候选截断"，是效果上限的对照）

要看两个维度：
    效果：Hit@1 / MRR 随 TOP_N 怎么变，在哪个点饱和
    成本：每题要送 TOP_N 个「问题+块」进模型，全评测集 = 25 × TOP_N 个句子
          cross-encoder 的成本是随候选数线性涨的，所以"效果饱和点"就是最优解。

用法：
    D:\\python3147\\python.exe eval\\topn_sweep.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import rerank as rr                                              # noqa: E402
from hybrid import load_all, orders_for, rrf_fuse                 # noqa: E402
from dense_index import embed_texts                               # noqa: E402
from step2_embed import embed                                     # noqa: E402
from rerank import reranked_orders, metrics, rank_of              # noqa: E402

CANDIDATES = [3, 6, 10, 15]      # 想加档位就往这里加，比如 [1, 3, 6, 10, 15]


def main():
    tfidf_mat, idf, dense_mat, chunks, questions = load_all()
    n = len(questions)
    n_chunks = len(chunks)

    print(f"\n语料 {n_chunks} 块 · 评测集 {n} 题 · 四个档位 {CANDIDATES}")
    print(f"预计 rerank 请求 {n * len(CANDIDATES)} 次，要等 3–5 分钟，别以为卡住了")
    print("-" * 78)

    print("先算一次两路召回 + RRF 融合（只算一遍，后面四档复用，省 API 调用）")
    a_orders = orders_for(questions, lambda q: tfidf_mat @ embed(q, idf), n_chunks)
    b_orders = orders_for(questions, lambda q: dense_mat @ embed_texts([q])[0], n_chunks)
    h_orders = [rrf_fuse(a, b) for a, b in zip(a_orders, b_orders)]

    _, h_hit1, h_mrr = metrics(h_orders, questions, n_chunks)
    print(f"hybrid 基线：Hit@1 {h_hit1}/{n} ({h_hit1 / n:.0%}) · MRR {h_mrr:.3f}")

    rows = []
    detail = {}
    for top_n in CANDIDATES:
        rr.TOP_N = top_n                       # 改模块里的常量，reranked_orders 用的就是它
        orders = reranked_orders(questions, chunks, h_orders)
        _, hit1, mrr = metrics(orders, questions, n_chunks)
        rows.append((top_n, hit1, mrr, n * top_n))
        detail[top_n] = [rank_of(o, q["gold_chunks"]) for o, q in zip(orders, questions)]
        print(f"  TOP_N={top_n:<3} 跑完：Hit@1 {hit1}/{n} ({hit1 / n:.0%}) · MRR {mrr:.3f}")

    print("\n" + "=" * 78)
    print("效果 vs 成本")
    print("=" * 78)
    print(f"{'TOP_N':<8}{'覆盖全库':>10}{'Hit@1':>14}{'MRR':>9}{'精排文本对':>12}{'成本倍数':>10}")
    base_cost = n * CANDIDATES[0]
    for top_n, hit1, mrr, cost in rows:
        print(f"{top_n:<8}{f'{top_n / n_chunks:.0%}':>10}"
              f"{f'{hit1}/{n} ({hit1 / n:.0%})':>14}{mrr:>9.3f}"
              f"{cost:>12}{cost / base_cost:>9.1f}x")

    print("\n" + "-" * 78)
    print("逐题对比：TOP_N=3 丢首位、加到 6 才捞回的题（说明截断造成的损失）")
    print("-" * 78)
    r3, r6 = detail[CANDIDATES[0]], detail[6] if 6 in detail else detail[CANDIDATES[-1]]
    lost = [(q["id"], q["question"][:20], a, b) for q, a, b in zip(questions, r3, r6) if a != 1 and b == 1]
    if not lost:
        print("  没有——TOP_N=3 已经够全，再加候选只是白花钱")
    for qid, qs, a, b in lost:
        print(f"  id{qid:<3} {qs:<22} 名次 {a} → 1")

    print("\n" + "-" * 78)
    print("四档都没拿到首位的题（TOP_N 再大也救不回来 → 得靠别的招，比如换模型/改 prompt）")
    print("-" * 78)
    hard = [(q["id"], q["question"][:20]) for q, ranks in zip(questions, zip(*detail.values()))
            if all(r != 1 for r in ranks)]
    if not hard:
        print("  没有——只要候选给够，精排全都能排对")
    for qid, qs in hard:
        print(f"  id{qid:<3} {qs}")


if __name__ == "__main__":
    main()
