# llm-rag-qa · 基于 LangChain 的文档 RAG 问答系统

> 针对《淮安大学学生事务手册》构建检索增强问答（RAG）链路。目标：回答奖学金、学籍、档案等规章问题时**有原文依据、可溯源**，而不是让大模型凭记忆编造。

## 为什么做这个

大模型直接回答具体校规问题时容易产生幻觉（编造不存在的条款）。RAG 的思路是：先把文档切成语义块，提问时先检索出最相关的原文片段，再让模型**只基于检索到的原文**回答，并标注出处页码。

## 技术栈

Python 3 · LangChain · DeepSeek API

## 当前进展

- [x] DeepSeek API 调用打通，API Key 通过环境变量管理（不进代码库）
- [x] PDF 文档加载（PyPDFLoader）与语义切分（RecursiveCharacterTextSplitter）
- [x] 切分策略实测与分析
- [ ] 向量化入库与相似度检索
- [ ] 问答链路与引用溯源
- [ ] 构建评测集，对比不同 chunk 尺寸的检索效果

## 切分实测数据

| 指标 | 数值 |
|---|---|
| 语料 | 《淮安大学学生事务手册》7 页 · 约 5900 字 |
| 切分参数 | chunk_size=500 · chunk_overlap=50 |
| 切分结果 | 15 块 · 平均 393 字 · 最长 496 · 最短 69 |

**两个关键发现：**

1. **块不跨页**：PyPDFLoader 按页加载，切分在页内进行。好处是溯源精确到页，代价是跨页的长条款会被截断，后续计划通过合并页面后切分来对比验证。
2. **overlap 生效**：相邻块首尾有 50 字重叠，保证被切断的句子在下一个块里保留上文。实测第 2 块开头是第 1 块结尾的重复句，说明 overlap 正常工作。

## 快速开始

```bash
git clone https://gitee.com/liujiahui-1/llm-rag-qa.git
cd llm-rag-qa
pip install langchain langchain-community pypdf requests python-dotenv
python step1_split.py
```

API Key 放在项目根目录 `.env` 文件中（已被 `.gitignore` 排除，不会上传）：

```
DEEPSEEK_API_KEY=sk-xxxx
```

## 项目结构

```
llm-rag-qa/
├── call_llm.py      # DeepSeek API 基础调用
├── step1_split.py   # PDF 加载与文本切分
├── data/            # 示例语料（学生事务手册）
└── output/          # 切分结果（本地生成，不入库）
```
