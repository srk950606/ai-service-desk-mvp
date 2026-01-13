import os
import httpx
from typing import List, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from pgvector import Vector

from openai import OpenAI

# 直接使用提供的API密钥
OPENAI_API_KEY = ""
# 设置代理配置
client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=30.0,
    base_url="https://api.openai.com/v1",
    http_client=httpx.Client(
        proxy="http://127.0.0.1:33210",
    )
)

PG_DSN = os.environ["PG_DSN"]  # e.g. "postgresql://user:pass@localhost:5432/kb"

app = FastAPI(title="KB RAG MVP")

# ---------- utils ----------
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        chunks.append(text[i:j])
        i = j - overlap
        if i < 0:
            i = 0
        if j == len(text):
            break
    return chunks

def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[Vector]:
    # Embeddings API: input 可以是数组（批量）
    resp = client.embeddings.create(model=model, input=texts)
    return [Vector(d.embedding) for d in resp.data]

def db_conn():
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    register_vector(conn)
    return conn

# ---------- API models ----------
class IngestReq(BaseModel):
    file_path: str
    title: str | None = None

class AskReq(BaseModel):
    question: str
    top_k: int = 6

# ---------- endpoints ----------
@app.post("/ingest")
def ingest(req: IngestReq):
    # 1) 读本地文件（MVP只处理txt/md）
    with open(req.file_path, "r", encoding="utf-8") as f:
        text = f.read()

    parts = chunk_text(text)
    if not parts:
        return {"ok": False, "reason": "empty file"}

    # 2) embedding
    vectors = embed_texts(parts)

    # 3) upsert document + insert chunks
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents(path, title)
                VALUES (%s, %s)
                ON CONFLICT (path) DO UPDATE SET title=EXCLUDED.title, updated_at=now()
                RETURNING doc_id
                """,
                (req.file_path, req.title or os.path.basename(req.file_path)),
            )
            doc_id = cur.fetchone()[0]

            rows = [(doc_id, idx, parts[idx], vectors[idx]) for idx in range(len(parts))]
            execute_values(
                cur,
                """
                INSERT INTO chunks(doc_id, chunk_index, content, embedding)
                VALUES %s
                """,
                rows,
                template="(%s, %s, %s, %s)",
            )

        return {"ok": True, "doc_id": doc_id, "chunks": len(parts)}
    finally:
        conn.close()


@app.post("/ask")
def ask(req: AskReq):
    # 1) embed question
    qvec = embed_texts([req.question])[0]

    # 2) vector search topK（cosine距离：越小越相似）
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.chunk_id, d.title, d.path, c.chunk_index, c.content,
                       (c.embedding <=> %s) AS distance
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (qvec, qvec, req.top_k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return {"answer": "知识库中没有检索到相关内容。", "citations": []}

    # 3) build context with citations
    evidence_blocks = []
    citations = []
    for (chunk_id, title, path, chunk_index, content, dist) in rows:
        tag = f"[{title}#chunk{chunk_index}]"
        evidence_blocks.append(f"{tag}\n{content}")
        citations.append({
            "title": title,
            "path": path,
            "chunk_index": chunk_index,
            "chunk_id": chunk_id,
            "distance": float(dist),
        })

    context = "\n\n---\n\n".join(evidence_blocks)

    # 4) generate answer using Responses API
    # （Responses API 是 OpenAI Cookbook 推荐的编排入口之一）
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "你是企业内部知识库助手。只能依据提供的“证据块”回答。\n"
                    "要求：\n"
                    "1) 如果证据不足，明确说明无法从文档得出结论，并给出你看到了哪些相关证据块。\n"
                    "2) 给出条理化答案。\n"
                    "3) 关键结论后用引用标签，如 [文档名#chunkX]。\n"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{req.question}\n\n证据块：\n{context}",
            },
        ],
    )

    answer_text = resp.output_text  # SDK 通常提供 output_text 便捷字段

    return {"answer": answer_text, "citations": citations}