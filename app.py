# -*- coding: utf-8 -*-
"""app.py —— 把 RAG 链路封装成 HTTP 服务（FastAPI）

启动（在 D:\\llm-rag-qa 目录下）：
    D:\\python3147\\python.exe -m uvicorn app:app --reload --port 8000
打开：
    http://127.0.0.1:8000

为什么需要这一步：
    前三个脚本只能在命令行用，一次一个人。封成 HTTP 服务之后：
      · 浏览器能直接访问（别人也能用，不用装 Python）
      · 手机、小程序、任何语言都能调
      · 可以同时来多个请求（谁提问不用排队）

四个接口：
    GET  /              单页前端
    POST /api/ask       提问，一次性返回完整答案（方便 curl 测试）
    POST /api/ask/stream 提问，边生成边返回（前端用这个，字一个个蹦出来）
    POST /api/upload    上传 PDF -> 重新切分 + 重建向量库
    GET  /api/health    健康检查：现在库里有多少块、来自哪个文件
"""

import os
import sys
import json
import shutil
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))   # 让 import 找到同目录的 step1/step2/step3

# 全部复用已有代码：服务层只负责"接请求、转调用、还结果"，不重复实现业务
from step1_split import DATA_DIR, split_pdf, save_chunks
from step2_embed import CHUNK_FILE, build as build_vectors
from step3_ask import build_prompt, retrieve

load_dotenv(ROOT / ".env")     # 显式传路径：从别的目录启动也能读到密钥

app = FastAPI(title="校园学生事务 RAG 问答", version="1.0")

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
SYSTEM_PROMPT = "你是学校学生事务问答助手，回答必须严谨、简短、有出处。"


class AskIn(BaseModel):
    """请求体：POST /api/ask 时前端发过来的 JSON 就是这个形状。
    有了它 FastAPI 会做类型校验，还会自动生成 /docs 接口文档。"""
    question: str
    top_k: int = 5


def kb_state():
    """当前知识库状态：块数、来源文件、平均块长。前端顶部的状态条用这个。"""
    if not CHUNK_FILE.exists():
        return {"chunks": 0, "source": None}
    chunks = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    if not chunks:
        return {"chunks": 0, "source": None}
    return {
        "chunks": len(chunks),
        "source": chunks[0].get("source"),
        "avg_len": sum(len(c["text"]) for c in chunks) // len(chunks),
    }


def api_key():
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        # 报清晰的 500，而不是让 requests 抛一个看不懂的 401
        raise HTTPException(500, "服务端没有配置 DEEPSEEK_API_KEY，请检查 .env")
    return key


@app.get("/", response_class=HTMLResponse)
def index():
    """首页：把 templates/index.html 原样返回给浏览器。"""
    html = (ROOT / "templates" / "index.html")
    if not html.exists():
        return HTMLResponse("<h1>缺少 templates/index.html</h1>", status_code=500)
    return HTMLResponse(html.read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    return {"status": "ok", **kb_state()}


@app.post("/api/ask")
def ask(body: AskIn):
    """提问 -> 检索 top_k 块 -> 拼 Prompt -> 调模型 -> 一次性返回"""
    if not body.question.strip():
        raise HTTPException(400, "问题不能为空")

    hits, scores = retrieve(body.question, body.top_k)

    resp = requests.post(
        DEEPSEEK_URL,
        headers={"Authorization": f"Bearer {api_key()}"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(body.question, list(zip(hits, scores)))},
            ],
            "temperature": 0.1,      # 事实问答锁低温：同一问题应该得到同一个答案
        },
        timeout=60,
    )
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"]

    return {
        "answer": answer,
        "sources": [
            {"page": c["page"], "chunk_id": c["id"], "score": round(s, 3), "preview": c["text"][:80]}
            for c, s in zip(hits, scores)
        ],
    }


def stream_model(prompt):
    """边生成边往外吐字。
    模型返回的是 SSE 流：每行形如 data: {"choices":[{"delta":{"content":"奖"}}]}
    我们把每一小块文字 yield 出去，前端收到就立刻追加到屏幕上。"""
    with requests.post(
        DEEPSEEK_URL,
        headers={"Authorization": f"Bearer {api_key()}"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "stream": True,          # 关键：打开流式
        },
        stream=True,
        timeout=60,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0].get("delta", {}).get("content", "")
            if delta:
                yield delta


def sse(obj):
    """打包成 SSE 格式：data: {json}\\n\\n（两个换行是消息分隔符，少一个浏览器就收不到）"""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.post("/api/ask/stream")
def ask_stream(body: AskIn):
    if not body.question.strip():
        raise HTTPException(400, "问题不能为空")

    hits, scores = retrieve(body.question, body.top_k)
    prompt = build_prompt(body.question, list(zip(hits, scores)))
    sources = [
        {"page": c["page"], "chunk_id": c["id"], "score": round(s, 3), "preview": c["text"][:80]}
        for c, s in zip(hits, scores)
    ]

    def gen():
        yield sse({"sources": sources})      # 先告诉前端答案来自哪几块
        try:
            for piece in stream_model(prompt):
                yield sse({"delta": piece})
        except Exception as e:                # 出错也要让前端知道，别一直转圈
            yield sse({"error": f"生成失败：{e}"})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """上传 PDF -> 落盘 -> 重新切分 -> 重建向量库。
    整条链路跑完才返回，所以文件大时会有几秒等待（真实产品会改成后台任务队列）。"""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "只支持 PDF 文件")

    DATA_DIR.mkdir(exist_ok=True)
    dest = DATA_DIR / Path(file.filename).name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)     # 分块写，避免大文件一次占满内存

    try:
        chunks, pages, empty_pages = split_pdf(dest)
    except ValueError as e:
        raise HTTPException(400, str(e))

    save_chunks(chunks)
    build_vectors()                          # 重建 15 行 × 1024 列的矩阵

    return {
        "file": dest.name,
        "pages": pages,
        "chunks": len(chunks),
        "avg_len": sum(len(c["text"]) for c in chunks) // len(chunks),
        "empty_pages": empty_pages,
        **kb_state(),
    }


if __name__ == "__main__":
    # 方便直接 python app.py 启动；等价于 uvicorn 命令行
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
