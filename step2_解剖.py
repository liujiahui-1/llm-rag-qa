# -*- coding: utf-8 -*-
"""
step2_解剖.py —— 把 step2_embed.py 拆开，一步一打印

这个文件不做检索，只做一件事：
拿三句短文本，把「文字 → 1024 个数字」的全过程一格一格打出来。
看懂这个，再回去看 step2_embed.py，就只是"同一条流水线 + 几个函数壳子"。
"""
import re
import hashlib
import numpy as np

DIM = 1024

A = "宿舍禁止使用热得快"
B = "国家奖学金八千元"
C = "宿舍内禁止使用大功率电器"


def show(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def cut(t):
    """切成特征：单字 + 相邻两字"""
    t = re.sub(r"\s+", "", t)
    return list(t) + [t[i:i + 2] for i in range(len(t) - 1)]


def pos(tok):
    """给一个特征算座位号：0 ~ 1023"""
    return int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16) % DIM


def vec(t):
    """把句子填进 1024 个格子"""
    v = np.zeros(DIM, dtype=np.float32)
    for tok in cut(t):
        v[pos(tok)] += 1
    return v


# ---------- 第 1 步 ----------
show("第 1 步：手里有三句话")
print("A：", A)
print("B：", B)
print("C：", C)
print("\n目标：判断 A 跟 B、C 谁更像。可计算机不认识汉字，只认识数字。")

# ---------- 第 2 步 ----------
show("第 2 步：切成「特征」（单字 + 相邻两字）")
fa, fc = cut(A), cut(C)
print(f"A 共 {len(fa)} 个特征：")
print("   ", " ".join(fa))
print(f"\nC 共 {len(fc)} 个特征：")
print("   ", " ".join(fc))
print("\n为什么要切两个字？因为「热得」和「热得快」意思差很远，")
print("只切单字的话，「热」「得」「快」三个字会跟「快件」「得分」混在一起。")

# ---------- 第 3 步 ----------
show("第 3 步：每个特征算出一个「座位号」（0~1023）")
print("算法：对这个词做 md5，取前 8 位十六进制，再除以 1024 取余数")
print()
for tok in fa[:7]:
    print(f"   「{tok}」 → 第 {pos(tok):>4} 号格子")
print("   ...（剩下的同理）")
print("\n座位号是「算出来」的，不是事先排好的表。")
print("好处：不管遇到什么新词，都能立刻算出位置，不用维护字典。")
print("代价：两个不同的词可能算到同一个格子（哈希冲突）—— 格子越多冲突越少。")

# ---------- 第 4 步 ----------
show("第 4 步：把句子填进 1024 个格子（这就是「向量」）")
va, vb, vc = vec(A), vec(B), vec(C)
for name, v, txt in [("A", va, A), ("B", vb, B), ("C", vc, C)]:
    nz = np.nonzero(v)[0]
    print(f"\n句子{name}（{txt}）：碰到 {len(nz)} 个格子，其余 {DIM - len(nz)} 个是 0")
    print("   ", "  ".join(f"[{i}]{v[i]:.0f}" for i in nz[:6]), " ...")

print("\n注意：这 1024 个数字里，绝大多数是 0，只有三百来个有值。")
print("所谓「文本向量化」，就是把一句话变成这样一行数字。")
print("真实模型（bge/m3）也是 1024 或 768 个数字，区别是它的数字是模型「学」出来的，")
print("我们这里的数字是「数每个词出现几次」数出来的。")

# ---------- 第 5 步 ----------
show("第 5 步：比相似度（点积 = 同格相乘再相加）")


def compare(n1, v1, n2, v2):
    both = np.nonzero(v1 * v2)[0]
    dot = float(v1 @ v2)
    print(f"\n{n1} × {n2}：")
    if len(both) == 0:
        print("   没有任何一个格子被两边同时碰到")
        print(f"   点积 = 0")
    else:
        for i in both[:6]:
            print(f"   第 {i} 格：{v1[i]:.0f} × {v2[i]:.0f} = {v1[i] * v2[i]:.0f}")
        if len(both) > 6:
            print(f"   ...共 {len(both)} 个格子重叠")
        print(f"   点积 = {dot:.1f}")
    cos = dot / (np.linalg.norm(v1) * np.linalg.norm(v2))
    print(f"   余弦相似度 = {dot:.1f} / ({np.linalg.norm(v1):.2f} × {np.linalg.norm(v2):.2f}) = {cos:.3f}")
    return cos


show("A 跟 B 比（毫无关系的两句话）")
compare("A", va, "B", vb)
show("A 跟 C 比（都在讲宿舍电器）")
compare("A", va, "C", vc)

# ---------- 结论 ----------
show("看完这五步，回头看 step2_embed.py")
print("""
    tokenize()   →  第 2 步：切特征
    hashed()     →  第 3 步：算座位号
    embed()      →  第 4 步：填格子（外加两项调整，见下）
    search()     →  第 5 步：比相似度，取前 3
    build()      →  把第 2~4 步对 15 个块各做一遍，存成矩阵

那两个「外加调整」是：
  ① 乘 IDF：满篇都有的词（学校/学生）权重压低，只在一处出现的词（热得快）权重抬高
             —— 不乘的话，谁都会因为「学生」这个词而看起来相关
  ② 除以长度（归一化）：500 字的块天然比 69 字的块数字大，
             不除的话「长」会被误当成「相关」

就这么两件事。剩下 80 行代码都是把这两件事写得更省事而已。
""")
