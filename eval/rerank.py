# -*- coding: utf-8 -*-
"""eval/rerank.py —— 在 hybrid（RRF）之后加一步精排：cross-encoder 重排

流水线：两路召回 → RRF 融合 → 取前 TOP_N 名 → bge-reranker 精排 → 最终名次

为什么 rerank 不能当检索器：
    bi-encoder（TF-IDF / bge-m3）把问题和块各自变成向量，可以提前建索引，检索时算得飞快；
    cross-encoder 必须把「问题+块」拼在一起现算，每对一个就要跑一次模型——
    10 万块全跑一遍天荒地老。所以它只精排融合后的前几名，快和准各归其位。

用法：
    D:\\python3147\\python.exe eval\\rerank.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from step2_embed import embed, VEC_FILE, CHUNK_FILE            # noqa: E402  后端 A
from dense_index import build_index, embed_texts               # noqa: E402  后端 B
from hybrid import load_all, orders_for, rank_of, rrf_fuse     # noqa: E402  复用昨天的混合检索

API_URL = "https://api.siliconflow.cn/v1/rerank"
MODEL = "BAAI/bge-reranker-v2-m3"
TOP_N = 6          # 送进精排的候选块数
SPARSE_LOST = {2, 4, 6, 16, 22, 24}   # 昨天稀疏丢首位的题
DENSE_LOST = {3, 10, 13, 21, 25}      # 昨天稠密丢首位的题


def _api_key() -> str:
    load_dotenv(ROOT / ".env")
    key = os.getenv("SILICONFLOW_API_KEY")
    if not key:
        raise SystemExit("缺少 SILICONFLOW_API_KEY，请先写进项目根目录的 .env")
    return key


def call_rerank(query: str, docs: list[str]) -> list[float]:
    """把「问题 + 候选块」送进 cross-encoder，返回每个块的相关性分数。

    参数:
        query: 用户的问题
        docs:  候选块原文列表（顺序和传入一致）

    返回:
        scores: 与 docs 等长的分数列表，scores[i] 是 docs[i] 的相关性分
                （接口返回顺序不保证，要按 index 字段归位——和 embed_texts 一样）

    接口约定:
        POST https://api.siliconflow.cn/v1/rerank
        请求体: {"model": MODEL, "query": query, "documents": docs}
        返回体: {"results": [{"index": 0, "relevance_score": 0.97}, ...]}
    """
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    resp = requests.post(
        API_URL,
        headers=headers,
        json={"model": MODEL, "query": query, "documents": docs},
        timeout=60
    )
    if resp.status_code != 200:
        raise SystemExit(f"rerank接口报错{resp.status_code}:{resp.text[:300]}")
    scores = [0.0] * len(docs)
    for item in resp.json()["results"]:
        scores[item["index"]] = item["relevance_score"]
    return scores


def reranked_orders(questions, chunks, hybrid_orders):
    """对每道题：取 hybrid 前 TOP_N 名送精排，后面的块保持原名次不动。"""
    out = []
    for q, order in zip(questions, hybrid_orders):
        cand = order[:TOP_N]                    # 前 6 名进精排
        rest = order[TOP_N:]                    # 第 7 名以后不参与
        docs = [chunks[i]["text"] for i in cand]
        scores = call_rerank(q["question"], docs)
        best_first = [cand[i] for i in np.argsort(scores)[::-1]]
        out.append(best_first + rest)
    return out


def metrics(orders, questions, k):
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
    r_orders = reranked_orders(questions, chunks, h_orders)

    print("\n" + "=" * 86)
    print(f"rerank 精排   语料 {n_chunks} 块 · 评测集 {n} 题 · 精排候选 {TOP_N} 名 · 模型 {MODEL}")
    print("=" * 86)
    print(f"{'指标':<10}{'TF-IDF':>12}{'bge-m3':>12}{'hybrid':>12}{'+rerank':>12}{'vs hybrid':>14}")
    for name, k in (("Hit@3", 3), ("Hit@1", n_chunks)):
        a = metrics(a_orders, questions, k)
        b = metrics(b_orders, questions, k)
        h = metrics(h_orders, questions, k)
        r = metrics(r_orders, questions, k)
        idx = 1 if name == "Hit@1" else 0
        print(f"{name:<10}{f'{a[idx]}/{n} ({a[idx]/n:.0%})':>12}"
              f"{f'{b[idx]}/{n} ({b[idx]/n:.0%})':>12}"
              f"{f'{h[idx]}/{n} ({h[idx]/n:.0%})':>12}"
              f"{f'{r[idx]}/{n} ({r[idx]/n:.0%})':>12}"
              f"{f'{r[idx]-h[idx]:+d} 题':>14}")
    m = [metrics(o, questions, n_chunks)[2] for o in (a_orders, b_orders, h_orders, r_orders)]
    print(f"{'MRR':<10}{m[0]:>12.3f}{m[1]:>12.3f}{m[2]:>12.3f}{m[3]:>12.3f}{m[3]-m[2]:>+14.3f}")

    print("\n" + "-" * 86)
    print("逐题名次（只看 +rerank 没拿第一，或昨天失守的题）")
    print("-" * 86)
    print(f"{'id':<4}{'题':<24}{'TF':>5}{'稠密':>6}{'hybrid':>7}{'+rerank':>8}   说明")
    for a, b, h, r, q in zip(a_orders, b_orders, h_orders, r_orders, questions):
        ra, rb = rank_of(a, q["gold_chunks"]), rank_of(b, q["gold_chunks"])
        rh, rr = rank_of(h, q["gold_chunks"]), rank_of(r, q["gold_chunks"])
        if rr == 1 and ra == 1 and rb == 1:
            continue
        qid = q["id"]
        tags = []
        if qid in SPARSE_LOST:
            tags.append("稀疏失守")
        if qid in DENSE_LOST:
            tags.append("稠密失守")
        if rh != 1 and rr == 1:
            tags.append("✅ rerank 捞回")
        elif rr == 1 and rh == 1 and min(ra, rb) != 1:
            tags.append("✅ 精排巩固")
        elif rr > rh:
            tags.append("⚠️ rerank 更差")
        print(f"{qid:<4}{q['question'][:22]:<24}{ra:>5}{rb:>6}{rh:>7}{rr:>8}   {'/'.join(tags)}")


if __name__ == "__main__":
    main()
