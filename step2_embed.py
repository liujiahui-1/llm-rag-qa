# -*- coding: utf-8 -*-
"""
step2_embed.py —— 文本块向量化 + 相似度检索

用法：
    python step2_embed.py                   # 建库：把 15 个块转成向量存起来
    python step2_embed.py "奖学金怎么评"     # 检索：返回最相关的 3 块

为什么不用 chromadb：
    向量库的本质只有两件事——把文本变成向量存起来、用的时候算相似度取前 k 个。
    这台机器的 Python 是 3.14，chromadb 依赖的 onnxruntime 还没出对应安装包，
    所以先用 numpy 把这套机制写出来；以后换 bge-m3 或 chroma，只需替换这两个环节。
"""
import sys
import json
import re
import hashlib
import numpy as np
from pathlib import Path

DIM = 1024  # 向量维度：越高哈希冲突越少、区分度越好，代价是存储和计算变大
ROOT = Path(__file__).parent
CHUNK_FILE = ROOT / "output" / "chunks_500_50.json"
VEC_FILE = ROOT / "output" / "vectors_500_50.npz"


def tokenize(text):
    """中文没有空格，这里用「单字 + 相邻两字」当特征。
    单字抓词根（奖/学/金），两字抓搭配（奖学/学金），
    比只切单字更能区分「奖学金」和「助学金」这种一字之差的概念。"""
    t = re.sub(r"\s+", "", text)
    return list(t) + [t[i:i + 2] for i in range(len(t) - 1)]


def hashed(token):
    """把任意长度的 token 映射到 0 ~ DIM-1 的固定槽位。
    用 md5 而不是内置 hash()：内置 hash 每次进程启动都换随机盐，
    存下来的向量下次启动就对不上了，检索会全乱。"""
    return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % DIM


def build_idf(texts):
    """IDF：一个词在越多块里出现，区分度越低，权重越小。
    「学校」几乎每块都有，靠它分不出哪块相关；「热得快」只在宿舍那块出现，一说就该命中。
    公式 log(N / (1+df)) + 1，加 1 是为了让"只出现在一块"的词权重不被压成 0。"""
    n = len(texts)
    df = np.zeros(DIM, dtype=np.float32)
    for t in texts:
        for h in {hashed(tok) for tok in tokenize(t)}:  # 先 set 去重：同一块里出现 10 次也只算 1 次
            df[h] += 1
    return np.log(n / (1 + df)) + 1


def embed(text, idf):
    """TF-IDF 加权后做 L2 归一化。
    归一化是必须的：500 字的块向量天然比 69 字的块大，不归一化的话
    「长」会被误当成「相关」。归一化之后，两个向量的点积就等于余弦相似度。"""
    v = np.zeros(DIM, dtype=np.float32)
    for tok in tokenize(text):
        v[hashed(tok)] += 1.0
    v = np.log1p(v) * idf          # 亚线性 TF：某个词重复 20 次不该是出现 1 次的 20 倍权重
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def build():
    chunks = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    idf = build_idf(texts)
    mat = np.vstack([embed(t, idf) for t in texts])
    np.savez(VEC_FILE, vectors=mat, idf=idf, ids=[c["id"] for c in chunks])
    print(f"已向量化 {len(chunks)} 块，矩阵形状 {mat.shape}（块数 × 维度）")
    print(f"每块变成 {DIM} 个数字，{len(chunks)} 块堆成一个 {mat.shape[0]}×{mat.shape[1]} 的矩阵")
    print(f"向量已保存：{VEC_FILE.relative_to(ROOT)}")
    print(f"平均每块非零维数：{(mat > 0).sum(axis=1).mean():.0f} / {DIM}")


def search(query, k=3):
    if not VEC_FILE.exists():
        sys.exit("还没建库，先不带参数跑一次：python step2_embed.py")
    data = np.load(VEC_FILE, allow_pickle=True)
    mat, idf = data["vectors"], data["idf"]
    chunks = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))

    q = embed(query, idf)
    scores = mat @ q                               # 两边都归一化过，点积 = 余弦相似度
    top = np.argsort(scores)[::-1][:k]

    print(f"\n查询：{query}")
    print("-" * 62)
    for rank, i in enumerate(top, 1):
        c = chunks[i]
        print(f"[{rank}] 相似度 {scores[i]:.3f} | 第 {c['page']} 页 | 块 {c['id']}")
        print("     " + c["text"][:80].replace("\n", " ") + " ...")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        search(" ".join(sys.argv[1:]))
    else:
        build()
