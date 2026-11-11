"""第一步：把 PDF 读进来，切成小块，存成 JSON 给后面的向量化用。

用法：
    python step1_split.py 你的文件.pdf
    python step1_split.py            # 不给参数时，自动用 data/ 目录下的第一个 pdf

为什么要有这一步：
    大模型的上下文放不下整本书，检索也只能按"块"去匹配，
    所以必须先切分。块太大 → 检索不准、浪费 token；
    块太小 → 上下文断裂、答非所问。chunk_size / chunk_overlap 就是要调的参数。
"""

import json
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- 1. 找到要处理的 PDF ----------
if len(sys.argv) > 1:
    pdf_path = Path(sys.argv[1])
else:
    candidates = sorted(Path("data").glob("*.pdf"))
    if not candidates:
        sys.exit("请把 PDF 放进 data/ 目录，或运行：python step1_split.py 你的文件.pdf")
    pdf_path = candidates[0]

if not pdf_path.exists():
    sys.exit(f"文件不存在：{pdf_path}")

print(f"读取文件：{pdf_path}")

# ---------- 2. 加载：PDF → 一页一个 Document ----------
docs = PyPDFLoader(str(pdf_path)).load()
print(f"页数：{len(docs)}")

# 有些 PDF 是扫描件，读出来是空字符串，这里先体检一下
empty_pages = [d.metadata.get("page", i) + 1 for i, d in enumerate(docs) if not d.page_content.strip()]
if empty_pages:
    print(f"警告：第 {empty_pages[:10]} 页没有提取到文字（可能是扫描版或纯图片 PDF）")
if sum(len(d.page_content.strip()) for d in docs) < 200:
    sys.exit("几乎没读到文字，换一份文本型 PDF 再试")

# ---------- 3. 切分：长文 → 小块 ----------
CHUNK_SIZE = 500      # 每块最多 500 个字符
CHUNK_OVERLAP = 50    # 相邻块重叠 50 个字，防止答案正好卡在切口处被切断

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    # 优先按段落、换行、句号切，实在切不开才硬切，这样块边界更自然
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)
chunks = splitter.split_documents(docs)
print(f"切出块数：{len(chunks)}（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）")

# ---------- 4. 体检：块的长度分布 ----------
lengths = [len(c.page_content) for c in chunks]
print(f"块长度：最短 {min(lengths)} / 平均 {sum(lengths) // len(lengths)} / 最长 {max(lengths)}")

# ---------- 5. 落盘：给下一步向量化用 ----------
out = Path("output")
out.mkdir(exist_ok=True)
out_file = out / f"chunks_{CHUNK_SIZE}_{CHUNK_OVERLAP}.json"
out_file.write_text(
    json.dumps(
        [
            {"id": i, "text": c.page_content, "page": c.metadata.get("page", -1) + 1, "source": pdf_path.name}
            for i, c in enumerate(chunks)
        ],
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(f"已保存：{out_file}")

# ---------- 6. 抽两块看看切得好不好 ----------
print("\n--- 第 1 块 ---")
print(chunks[0].page_content[:300])
print("\n--- 中间某块 ---")
mid = len(chunks) // 2
print(chunks[mid].page_content[:300])

print("\n检查标准：每块读起来应该是完整的一两句，而不是半句话断在开头。")
