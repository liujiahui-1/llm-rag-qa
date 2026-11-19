"""稠密索引：调用 SiliconFlow 的 bge-m3 把文本变成向量，缓存成 npz。

与 step2_embed.py（手写 TF-IDF 稀疏向量）的区别只在于「怎么把文本变成向量」这一步：
  - step2_embed：数词、算 idf、哈希投币 → 1024 维稀疏向量
  - 本文件   ：把文本发给 bge-m3，模型返回 1024 维语义向量

其余部分（余弦相似度、取前 k 个）完全一样——这就是「换掉一个函数，管线不动」。
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

API_URL = "https://api.siliconflow.cn/v1/embeddings"
MODEL = "BAAI/bge-m3"

CHUNK_FILE = ROOT / "output" / "chunks_500_50.json"
DENSE_FILE = ROOT / "output" / "dense_bge_m3_500_50.npz"


def _api_key() -> str:
    load_dotenv(ROOT / ".env")
    key = os.getenv("SILICONFLOW_API_KEY")
    if not key:
        raise SystemExit("缺少 SILICONFLOW_API_KEY，请先写进项目根目录的 .env")
    return key


def embed_texts(texts: list[str], batch_size: int = 16, verbose: bool = False) -> np.ndarray:
    """把一批文本变成向量矩阵，形状 (len(texts), 1024)。"""
    key = _api_key()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    out: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = requests.post(
            API_URL,
            headers=headers,
            json={"model": MODEL, "input": batch, "encoding_format": "float"},
            timeout=120,
        )
        if resp.status_code != 200:
            raise SystemExit(f"接口报错 {resp.status_code}: {resp.text[:300]}")
        # 返回顺序不保证，按 index 字段归位
        items = sorted(resp.json()["data"], key=lambda x: x["index"])
        out.extend(item["embedding"] for item in items)
        if verbose:
            print(f"  已向量化 {min(start + batch_size, len(texts))}/{len(texts)} 条")

    mat = np.array(out, dtype=np.float32)
    # 逐行做 L2 归一化，这样点积就等于余弦相似度（与 step2_embed 保持一致）
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _fingerprint(chunks: list[dict]) -> str:
    """语料指纹：切块内容一变，缓存就作废、自动重建。"""
    raw = json.dumps([c["text"] for c in chunks], ensure_ascii=False)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_index(force: bool = False) -> tuple[np.ndarray, list[dict]]:
    chunks = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    fp = _fingerprint(chunks)

    if DENSE_FILE.exists() and not force:
        data = np.load(DENSE_FILE, allow_pickle=True)
        if str(data["fingerprint"]) == fp:
            print(f"命中缓存：{DENSE_FILE.name}（{data['matrix'].shape[0]} 块 × {data['matrix'].shape[1]} 维）")
            return data["matrix"], chunks

    print(f"正在调用 {MODEL} 向量化 {len(chunks)} 个块…")
    mat = embed_texts([c["text"] for c in chunks], verbose=True)
    np.savez_compressed(DENSE_FILE, matrix=mat, fingerprint=fp)
    print(f"已保存：{DENSE_FILE.name}（{mat.shape[0]} 块 × {mat.shape[1]} 维）")
    return mat, chunks


if __name__ == "__main__":
    m, ch = build_index(force="--rebuild" in os.sys.argv)
    print("块数:", len(ch), "| 矩阵:", m.shape)
