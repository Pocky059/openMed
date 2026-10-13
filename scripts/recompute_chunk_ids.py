"""
v2 阶段 1 第 3 步：离线重算 chunk 索引（chunk_index.json）。

为什么需要它：评测判分靠 chunk id 精确匹配「模型检索到了哪一片」，
而 ChromaDB 里的 chunk id = md5(title_i_chunk[:50]) 依赖入库时的切片结果。
判分器不该连 ChromaDB（离线可测、零依赖），所以把切片与 id 计算在本脚本
离线重算一遍，产出 data/eval/chunk_index.json：
  - 问答集出题时，gold_chunks 对着 chunks[].text 选金标准片段；
  - 判分时，gold_chunks / 检索结果的 chunk id 直接查这个索引。

一致性铁律：切片函数与 md5 公式与 mcp/knowledge_base.py 的
KnowledgeBase._chunk_text（L369）和 add_documents（L182）逐字一致——
这里不复用 import 是因为 mcp.knowledge_base 的 import 链会拉起 chromadb
与 embedding 工厂（重且受 HF 离线模式约束），判分路径不值得背这条链。
任何一处改动必须双向同步（validate_exams 的 chunk 可复算断言会拦住漂移）。

运行时机：每次重新入库（scripts/load_knowledge.py）之后跑一次本脚本。
用法：
    .venv/Scripts/python.exe scripts/recompute_chunk_ids.py
"""
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CORPUS_FILE = ROOT / "data" / "knowledge" / "multihop_corpus.jsonl"
HOSPITAL_DIR = ROOT / "data" / "knowledge" / "hospital"
OUT_FILE = ROOT / "data" / "eval" / "chunk_index.json"


def chunk_text(text: str, chunk_size: int = 500):
    """按 chunk_size 切片，保留语义完整性（按句号/换行切分）。

    ⚠️ 与 mcp/knowledge_base.py 的 KnowledgeBase._chunk_text（L369）逐字一致。
    改动须双向同步（注释双向互指）。
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    current = ""
    # 按句子切分
    sentences = text.replace("\n", "。").split("。")
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 > chunk_size:
            if current:
                chunks.append(current)
            current = sent
        else:
            current = f"{current}。{sent}" if current else sent

    if current:
        chunks.append(current)

    return chunks


def chunk_id_of(title: str, index: int, text: str) -> str:
    """⚠️ 与 mcp/knowledge_base.py 的 KnowledgeBase.add_documents（L182）逐字一致。"""
    return hashlib.md5(f"{title}_{index}_{text[:50]}".encode()).hexdigest()


def build_doc_entry(title: str, doc_type: str, source_file: str, content: str) -> dict:
    """切一篇文档，产出 {title, doc_type, source_file, chunks:[{id, index, text}]}。"""
    chunks = []
    for i, chunk in enumerate(chunk_text(content, chunk_size=500)):
        chunks.append({"id": chunk_id_of(title, i, chunk), "index": i, "text": chunk})
    return {"title": title, "doc_type": doc_type, "source_file": source_file, "chunks": chunks}


def main() -> None:
    docs: dict = {}
    seen_keys: set = set()

    # 1. 多跳语料：键 = doc_id（全局唯一，判分用 doc_id 精确查）
    corpus_rows = [
        json.loads(line) for line in CORPUS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    for row in corpus_rows:
        entry = build_doc_entry(row["title"], row["doc_type"], CORPUS_FILE.name, row["content"])
        docs[row["doc_id"]] = entry
        seen_keys.add(row["doc_id"])

    # 2. hospital 语料：无 doc_id，键 = title（26 篇标题全局唯一；出题 source_docs 写文件名）
    hospital_files = sorted(HOSPITAL_DIR.glob("*.json"))
    for hf in hospital_files:
        for doc in json.loads(hf.read_text(encoding="utf-8")):
            title = doc["title"]
            if title in seen_keys:
                raise ValueError(f"hospital 文档标题与既有键冲突: {title}")
            docs[title] = build_doc_entry(title, "hospital", hf.name, doc["content"])
            seen_keys.add(title)

    total_chunks = sum(len(d["chunks"]) for d in docs.values())
    out = {
        "generated_at": date.today().isoformat(),
        # 语料版本说明：判分器/出题者据此核对「这批 chunk 是拿哪版语料切的」
        "source": {
            "multihop_corpus": f"{CORPUS_FILE.name} {len(corpus_rows)} 篇",
            "hospital_files": [f"{hf.name} ({len(json.loads(hf.read_text(encoding='utf-8')))} 篇)" for hf in hospital_files],
        },
        "doc_count": len(docs),
        "total_chunks": total_chunks,
        "docs": docs,
    }
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(docs)} 篇文档 / {total_chunks} 片段 -> {OUT_FILE.name}")


if __name__ == "__main__":
    main()
