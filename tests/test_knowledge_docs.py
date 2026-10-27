"""知识库文档列表测试（2026-10-27 前端知识库管理页 GET /knowledge/docs）。

不连 ChromaDB：mock collection.get 返回 metadata，测聚合与内置/用户归类逻辑。
"""
import asyncio

import pytest

from mcp.knowledge_base import KnowledgeBase, _load_hospital_docs


class FakeCollection:
    """实现聚合/取文档用到的 get(include=[...])，documents 缺省时按 metadatas 长度补空串。"""

    def __init__(self, metadatas, documents=None):
        self.metadatas = metadatas
        self.documents = documents if documents is not None else [""] * len(metadatas)

    def get(self, include=None):
        return {"metadatas": self.metadatas, "documents": self.documents}


def make_kb(metadatas, documents=None):
    # 绕过 __init__（会真连 ChromaDB），只替换 _collection
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._collection = FakeCollection(metadatas, documents)
    return kb


def test_list_documents_aggregates_by_title_and_flags_builtin():
    """按标题聚合片段数；hospital 目录标题归 builtin，其余归 user。"""
    builtin_title = _load_hospital_docs()[0]["title"]
    kb = make_kb([
        {"title": builtin_title, "chunk_index": 0},
        {"title": builtin_title, "chunk_index": 1},
        {"title": "用户上传的须知", "chunk_index": 0},
        None,   # 脏 metadata 不崩（历史数据可能有 None 条目）
        {},
    ])
    data = kb.list_documents()
    assert data["total_chunks"] == 3
    by_title = {d["title"]: d for d in data["docs"]}
    assert by_title[builtin_title] == {"title": builtin_title, "chunks": 2, "source": "builtin"}
    assert by_title["用户上传的须知"] == {"title": "用户上传的须知", "chunks": 1, "source": "user"}
    assert data["builtin_count"] == 1
    assert data["user_count"] == 1


def test_list_documents_empty_kb():
    """空库返回零值结构，前端不用特判。"""
    kb = make_kb([])
    assert kb.list_documents() == {
        "docs": [],
        "total_chunks": 0,
        "builtin_count": 0,
        "user_count": 0,
    }


def test_get_document_returns_chunks_sorted_and_filtered():
    """按标题取片段：过滤无关标题（含脏 metadata），按 chunk_index 升序还原顺序。"""
    kb = make_kb(
        [
            {"title": "甲", "chunk_index": 1},
            {"title": "甲", "chunk_index": 0},
            {"title": "乙", "chunk_index": 0},
            None,
        ],
        documents=["甲片2", "甲片1", "乙片", "脏"],
    )
    assert kb.get_document("甲") == {
        "title": "甲",
        "chunks": [
            {"index": 0, "content": "甲片1"},
            {"index": 1, "content": "甲片2"},
        ],
    }


def test_get_document_unknown_title_returns_empty():
    """标题不存在 => 空 chunks，前端展示「该标题下没有片段」。"""
    kb = make_kb([{"title": "甲", "chunk_index": 0}], documents=["甲片"])
    assert kb.get_document("不存在") == {"title": "不存在", "chunks": []}


def test_knowledge_doc_endpoint(monkeypatch):
    """GET /knowledge/doc/{title} 端点：从工具管理器取 kb，按标题返回片段。"""
    import api.main as main

    class FakeKB:
        async def get_document_async(self, title):
            return {"title": title, "chunks": [{"index": 0, "content": "正文"}]}

    class FakeTool:
        def __init__(self):
            self.handler = FakeKB().get_document_async  # bound method：端点靠 handler.__self__ 取 kb

    class FakeToolManager:
        _tools = {"knowledge_search": FakeTool()}

    monkeypatch.setattr(main, "_tool_manager", FakeToolManager())
    data = asyncio.run(main.knowledge_doc("某药品说明书"))
    assert data["title"] == "某药品说明书"
    assert data["chunks"][0]["content"] == "正文"


def test_knowledge_docs_endpoint(monkeypatch):
    """GET /knowledge/docs 端点：从工具管理器取 kb 并返回文档列表。"""
    import api.main as main

    class FakeKB:
        async def list_documents_async(self):
            return {"docs": [{"title": "t", "chunks": 1, "source": "builtin"}],
                    "total_chunks": 1, "builtin_count": 1, "user_count": 0}

    class FakeTool:
        def __init__(self):
            self.handler = FakeKB().list_documents_async  # bound method：端点靠 handler.__self__ 取 kb

    class FakeToolManager:
        _tools = {"knowledge_search": FakeTool()}

    monkeypatch.setattr(main, "_tool_manager", FakeToolManager())
    data = asyncio.run(main.knowledge_docs())
    assert data["docs"][0]["title"] == "t"


def test_knowledge_docs_endpoint_503(monkeypatch):
    """工具未注册（知识库未初始化）=> 503。"""
    import api.main as main
    from fastapi import HTTPException

    class EmptyToolManager:
        _tools = {}

    monkeypatch.setattr(main, "_tool_manager", EmptyToolManager())
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.knowledge_docs())
    assert exc.value.status_code == 503
