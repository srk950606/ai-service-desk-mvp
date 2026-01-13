

---

## 目标 MVP（你要实现的最小闭环）

1. **ingest（入库）**：读取本地文件 → 分块（chunk）→ 调 Embeddings → 写入 Postgres(pgvector)
2. **ask（问答）**：问题 → Embeddings → pgvector 相似度检索 topK chunks → 组装上下文 → 调 **Responses API** 生成答案（附引用）

> Embedding 推荐 `text-embedding-3-small`（向量维度默认 1536）。([OpenAI 平台][1])
> pgvector 支持 HNSW/IVFFlat 索引；MVP 直接 HNSW 即可。([GitHub][2])

---

## 1) 数据库（Postgres + pgvector）

### 1.1 安装扩展 & 建表（SQL）

```sql
-- 1) 启用 pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2) 文档表（可极简）
CREATE TABLE IF NOT EXISTS documents (
  doc_id      BIGSERIAL PRIMARY KEY,
  path        TEXT UNIQUE NOT NULL,
  title       TEXT,
  updated_at  TIMESTAMPTZ DEFAULT now()
);

-- 3) 分块表：content + embedding
-- text-embedding-3-small 默认 1536 维
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id    BIGSERIAL PRIMARY KEY,
  doc_id      BIGINT REFERENCES documents(doc_id) ON DELETE CASCADE,
  chunk_index INT NOT NULL,
  content     TEXT NOT NULL,
  embedding   VECTOR(1536) NOT NULL
);

-- 4) 向量索引（HNSW + cosine）
-- 注：vector_cosine_ops 是 pgvector 常用 cosine 距离操作符族
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
ON chunks USING hnsw (embedding vector_cosine_ops);
```

pgvector 的索引类型与近邻检索能力（HNSW/IVFFlat）见官方说明。([GitHub][2])

---

## 2) 文档入库：最简 ingestion 方案

### 2.1 分块策略（MVP够用）

* 先只支持：`.txt / .md`（最快闭环）
* 分块：按字符长度粗切（例如 800～1200 中文字符/块）+ 少量 overlap（比如 100 字符）
* 后续再加：PDF/DOCX/PPTX 解析（不是 MVP 必需）

### 2.2 入库接口（FastAPI 示例）

> 说明：代码用 OpenAI **Embeddings API** + **Responses API**。Embeddings 的使用方式见官方指南。([AI文档][3])

```python
import os
from typing import List, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import execute_values

from openai import OpenAI

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

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

def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    # Embeddings API: input 可以是数组（批量）
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def db_conn():
    return psycopg2.connect(PG_DSN)

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
    conn.autocommit = True
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
        model="gpt-5.2",
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
```

> 上面代码里：
>
> * `text-embedding-3-small` 用于向量化文本（官方模型页与 embedding 指南）。([OpenAI 平台][1])
> * pgvector 的 HNSW/IVFFlat 索引能力见其项目说明。([GitHub][2])
> * RAG/Responses API 的整体用法可参考 OpenAI Cookbook 的示例思路。([OpenAI 厨房][4])

---

## 3) 你现在就能跑起来的最小步骤

1. 准备 Postgres，安装 pgvector 扩展（或用带扩展的镜像）
2. 执行上面的 SQL
3. 设置环境变量：

* `OPENAI_API_KEY=...`
* `PG_DSN=postgresql://...`

4. 启动服务：`uvicorn app:app --reload`
5. 先入库：`POST /ingest` 传 `file_path`
6. 再问答：`POST /ask` 传 `question`

---

## 4) MVP 之后最值得立刻加的 3 个增强（仍然很小）

1. **支持 PDF/DOCX/PPTX 解析**（提升知识覆盖）
2. **混合检索（BM25 + 向量）**（术语/编号类问题命中率大幅提升）
3. **Chunk 去重与同文档聚合**（避免 topK 都来自同一页）
