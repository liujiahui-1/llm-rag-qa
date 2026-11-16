"""评测集辅助工具：帮你在出题时找到「答案在第几块」

用法 1：通读全部 15 块，心里有地图
    D:\\python3147\\python.exe eval\\show_chunks.py

用法 2：按关键词查块——直接告诉你这个词出现在哪几块
    D:\\python3147\\python.exe eval\\show_chunks.py 奖学金
    D:\\python3147\\python.exe eval\\show_chunks.py 转专业
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "output" / "chunks_500_50.json"

def main():
    chunks = json.load(open(CHUNKS, encoding="utf-8"))

    if len(sys.argv) == 1:
        # 用法 1：全部打印，每块显示前 200 字
        print(f"共 {len(chunks)} 块\n")
        for c in chunks:
            text = c["text"].replace("\n", " ")
            print(f"[{c['id']}] {text[:200]}")
            print("-" * 60)
    else:
        # 用法 2：关键词查块
        keyword = sys.argv[1]
        print(f"关键词「{keyword}」出现在以下块中：\n")
        found = 0
        for c in chunks:
            if keyword in c["text"]:
                pos = c["text"].find(keyword)
                snippet = c["text"][max(0, pos - 30) : pos + 50].replace("\n", " ")
                print(f"[{c['id']}] ...{snippet}...")
                found += 1
        if found == 0:
            print("（没找到。换个词试试，比如：奖学金 / 贷款 / 绩点 / 实习 / 请假）")

if __name__ == "__main__":
    main()
