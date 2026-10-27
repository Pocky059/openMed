"""chat 端点兜底测试（2026-10-26：端点级 try/except 降级响应）。

直接调端点函数 + monkeypatch 全局 _chat_service，不起 FastAPI——
lifespan 会 assemble 全套组件（要连 Redis/ChromaDB/LLM），测试里起不来。
"""
import asyncio

import pytest


def test_chat_degrades_to_friendly_response(monkeypatch):
    """主链路抛异常时返回降级 ChatResponse，而不是裸 500。"""
    import api.main as main

    class BoomService:
        async def handle_chat_turn(self, *args, **kwargs):
            raise RuntimeError("llm down")

    monkeypatch.setattr(main, "_chat_service", BoomService())
    resp = asyncio.run(main.chat(main.ChatRequest(message="我胸口疼", user_id="u1")))
    assert resp.response == "系统暂时不可用，请稍后再试。"
    assert resp.intent == "other"       # 降级响应不虚构意图
    assert resp.agent_type == "none"    # 没有任何 agent 实际处理过
    assert resp.escalated is False
    assert resp.latency_ms == 0.0


def test_chat_503_when_service_not_ready(monkeypatch):
    """_chat_service 未初始化时报 503（如实报未就绪，不是 500）。"""
    import api.main as main
    from fastapi import HTTPException

    monkeypatch.setattr(main, "_chat_service", None)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.chat(main.ChatRequest(message="你好", user_id="u1")))
    assert exc.value.status_code == 503
