# -*- coding: utf-8 -*-
"""第一步：把 PDF 读进来，切成小块，存成 JSON 给后面的向量化用。

两种用法：
    ① 命令行：python step1_split.py 你的文件.pdf
    ② 被别的程序调用：from step1_split import split_pdf, save_chunks
       （app.py 上传 PDF 后要重新建库，走的就是这条）

为什么要有这一步：
    大模型的上下文放不下整本书，检索也只能按"块"去匹配，
    所以必须先切分。块太大 → 检索不准、浪费 token；
    块太小 → 上下文断裂、答非所问。chunk_size / chunk_overlap 就是要调的参数。

为什么用 pypdf 直接读、不用 langchain 的 PyPDFLoader：
    一是少一层依赖（PyPDFLoader 最终也是调 pypdf）；
    二是这台机器上有应用程序控制策略，langchain_community 的导入链会撞到
    被拦截的 xxhash DLL，直接 import 就报错。
    教训：依赖越薄，环境越不容易崩。
"""

import json
import sys
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "output"
DATA_DIR = ROOT / "data"

CHUNK_SIZE = 500      # 每块最多 500 个字符
CHUNK_OVERLAP = 50    # 相邻块重叠 50 个字，防止答案正好卡在切口处被切断


def split_pdf(pdf_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """PDF 文件 -> (块列表, 页数, 空页列表)

    块格式：{"id": 序号, "text": 正文, "page": 第几页, "source": 文件名}

    注意：按页逐个切，所以**块不会跨页**。
    好处是每条答案都能精确溯源到页码；代价是跨页的长条款会被切成两半。
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    page_texts = [(p.extract_text() or "") for p in reader.pages]

    if sum(len(t.strip()) for t in page_texts) < 200:
        raise ValueError("几乎没读到文字，可能是扫描版或纯图片 PDF")

    empty_pages = [i + 1 for i, t in enumerate(page_texts) if not t.strip()]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 优先按段落、换行、句号切，实在切不开才硬切，这样块边界更自然
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )

    chunks = []
    for page_no, text in enumerate(page_texts, start=1):   # 页码从 1 开始，和人对得上
        if not text.strip():
            continue
        for piece in splitter.split_text(text):
            chunks.append({
                "id": len(chunks),
                "text": piece,
                "page": page_no,
                "source": pdf_path.name,
            })

    return chunks, len(page_texts), empty_pages


def save_chunks(chunks, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """块列表落盘成 JSON，路径与 step2_embed.py 约定的一致（chunks_500_50.json）"""
    OUT_DIR.mkdir(exist_ok=True)
    out_file = OUT_DIR / f"chunks_{chunk_size}_{chunk_overlap}.json"
    out_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_file


def main():
    # ---------- 1. 找到要处理的 PDF ----------
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        candidates = sorted(DATA_DIR.glob("*.pdf"))
        if not candidates:
            sys.exit("请把 PDF 放进 data/ 目录，或运行：python step1_split.py 你的文件.pdf")
        pdf_path = candidates[0]
    if not pdf_path.exists():
        sys.exit(f"文件不存在：{pdf_path}")

    print(f"读取文件：{pdf_path}")

    # ---------- 2. 切分 ----------
    chunks, pages, empty_pages = split_pdf(pdf_path)
    print(f"页数：{pages}")
    if empty_pages:
        print(f"警告：第 {empty_pages[:10]} 页没有提取到文字（可能是扫描版或纯图片 PDF）")
    print(f"切出块数：{len(chunks)}（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）")

    # ---------- 3. 体检：块的长度分布 ----------
    lengths = [len(c["text"]) for c in chunks]
    print(f"块长度：最短 {min(lengths)} / 平均 {sum(lengths) // len(lengths)} / 最长 {max(lengths)}")

    # ---------- 4. 落盘 ----------
    out_file = save_chunks(chunks)
    print(f"已保存：{out_file}")

    # ---------- 5. 抽两块看看切得好不好 ----------
    print("\n--- 第 1 块 ---")
    print(chunks[0]["text"][:300])
    print("\n--- 中间某块 ---")
    print(chunks[len(chunks) // 2]["text"][:300])
    print("\n检查标准：每块读起来应该是完整的一两句，而不是半句话断在开头。")


if __name__ == "__main__":
    main()
