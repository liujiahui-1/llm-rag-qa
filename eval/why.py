"""badcase 诊断工具：打印问句与召回块的词级重叠，看检索为什么输。

用法：
    python eval/why.py 4          # 看评测集里 id=4 那道题
    python eval/why.py 宿舍几点关门  # 直接诊断任意一句话
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from step2_embed import tokenize, build_idf, hashed  # noqa: E402

VEC_FILE = ROOT / "output" / "vectors_500_50.npz"
CHUNK_FILE = ROOT / "output" / "chunks_500_50.json"
QUESTIONS = ROOT / "eval" / "eval_questions.json"


def load():
    data = np.load(VEC_FILE, allow_pickle=True)
    chunks = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    return chunks, data["idf"]


def show(query, gold, chunks, idf):
    q_tokens = tokenize(query)
    # 每个问句词的 idf 权重（越高说明越"稀有"、越能代表这句话）
    q_weights = {t: float(idf[hashed(t)]) for t in q_tokens}

    print(f"\n问句：{query}    正确块：{gold}")
    print("问句分词（括号里是该词的含金量 idf）：")
    print("   " + "  ".join(f"{t}({w:.1f})" for t, w in sorted(q_weights.items(), key=lambda x: -x[1])))
    print("-" * 66)

    # 算一遍真实排名
    from step2_embed import embed
    data = np.load(VEC_FILE, allow_pickle=True)
    v = embed(query, data["idf"])
    scores = data["vectors"] @ v
    top = np.argsort(scores)[::-1][:3]

    for rank, idx in enumerate(top, 1):
        idx = int(idx)
        mark = "✅" if idx in gold else "❌"
        text = chunks[idx]["text"]
        c_tokens = tokenize(text)
        # 问句里的词，哪些也出现在这个块里（命中=检索得分的来源）
        hit = [t for t in q_tokens if t in c_tokens]
        print(f"{mark} 第{rank}名 块{idx} 相似度{scores[idx]:.3f}  命中的词: {' '.join(hit) if hit else '(无)'}")
        print(f"      块开头: {text[:44]}".replace("\n", " "))
    print()


def main():
    chunks, idf = load()
    args = sys.argv[1:]
    if not args:
        sys.exit("用法：python eval/why.py <题目id 或 任意一句话>")
    arg = args[0]
    if arg.isdigit():
        qs = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
        q = next((x for x in qs if x["id"] == int(arg)), None)
        if q is None:
            sys.exit(f"评测集里没有 id={arg}")
        show(q["question"], q["gold_chunks"], chunks, idf)
    else:
        show(arg, [], chunks, idf)


if __name__ == "__main__":
    main()
