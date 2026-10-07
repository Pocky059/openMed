"""
v2 阶段 0.5 LLM 双协议客户端的单元测试。

覆盖 LLMProtocolAdapter 的两大职责：
1. 请求方向：Anthropic 风格 kwargs → OpenAI 请求体
   （system 注入 / 消息翻译 / tools / tool_choice / stop_sequences）；
2. 响应方向：OpenAI 响应 → Anthropic 风格 block 列表
   （text / tool_use / stop_reason 映射 / usage 改名）。

全部离线：FakeOpenAI 替身只记录请求、返回预设响应，不发起任何网络调用。
"""
import asyncio
import json
from types import SimpleNamespace

from anthropic import AsyncAnthropic

from core.llm_utils import (
    LLMProtocolAdapter,
    create_llm_client,
    extract_text_content,
)


# ── 替身 ──────────────────────────────────────────────────────────────────────

def make_response(*, content=None, tool_calls=None, finish_reason="stop",
                  with_usage=True):
    """构造一个 OpenAI 风格的响应替身（SimpleNamespace 模拟属性访问）。"""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7) if with_usage else None
    return SimpleNamespace(choices=[choice], usage=usage)


def make_tool_call(name, args_json, call_id="toolu_1"):
    """构造 OpenAI 风格的 tool_calls 条目替身。"""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=args_json),
    )


class FakeOpenAI:
    """替身 OpenAI 客户端：记录每次 create 收到的请求体，返回预设响应。"""

    def __init__(self, response):
        self.response = response
        self.requests = []   # 每次 create 的 kwargs 都记在这里供断言
        self.chat = SimpleNamespace(completions=self._Completions(self))

    class _Completions:
        def __init__(self, outer):
            self._outer = outer

        async def create(self, **kwargs):
            self._outer.requests.append(kwargs)
            return self._outer.response


def make_adapter(response) -> LLMProtocolAdapter:
    return LLMProtocolAdapter(FakeOpenAI(response))


def call(adapter, **kwargs):
    return asyncio.run(adapter.messages.create(**kwargs))


# ── 请求方向：Anthropic kwargs → OpenAI 请求体 ────────────────────────────────

def test_system_prompt_becomes_first_message():
    """OpenAI 没有 system 参数：system 文本必须变成消息列表头部的 system 角色消息。"""
    adapter = make_adapter(make_response(content="好的"))
    call(adapter, model="test-model", system="你是分诊助手",
         messages=[{"role": "user", "content": "我头疼"}])

    req = adapter._openai.requests[0]
    assert req["messages"][0] == {"role": "system", "content": "你是分诊助手"}
    assert req["messages"][1] == {"role": "user", "content": "我头疼"}


def test_plain_text_messages_pass_through():
    """纯文本消息原样透传；model/temperature/max_tokens 同名透传。"""
    adapter = make_adapter(make_response(content="ok"))
    call(adapter, model="test-model", temperature=0.3, max_tokens=512,
         messages=[{"role": "user", "content": "问题"}])

    req = adapter._openai.requests[0]
    assert req["model"] == "test-model"
    assert req["temperature"] == 0.3
    assert req["max_tokens"] == 512
    assert req["messages"] == [{"role": "user", "content": "问题"}]


def test_assistant_tool_use_blocks_become_tool_calls():
    """tool_use block → tool_calls 数组；input 序列化成 JSON 字符串，中文不转义。"""
    adapter = make_adapter(make_response(content=None, tool_calls=[], finish_reason="tool_calls"))
    call(adapter, model="test-model",
         messages=[{
             "role": "assistant",
             "content": [
                 {"type": "text", "text": "我先查一下知识库"},
                 {"type": "tool_use", "id": "toolu_1", "name": "search_knowledge_base",
                  "input": {"query": "银杏叶 华法林"}},
             ],
         }])

    req = adapter._openai.requests[0]
    msg = req["messages"][0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "我先查一下知识库"
    call_block = msg["tool_calls"][0]
    assert call_block["id"] == "toolu_1"
    assert call_block["function"]["name"] == "search_knowledge_base"
    assert json.loads(call_block["function"]["arguments"]) == {"query": "银杏叶 华法林"}
    assert "银杏叶" in call_block["function"]["arguments"], "中文应原样保留，不转 \\uXXXX"


def test_tool_result_blocks_become_tool_messages():
    """Anthropic 把工具结果塞在 user 消息里；OpenAI 要拆成 role=tool 的独立消息。"""
    adapter = make_adapter(make_response(content="查到了"))
    call(adapter, model="test-model",
         messages=[{
             "role": "user",
             "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                          "content": '[{"success": true}]'}],
         }])

    req = adapter._openai.requests[0]
    assert req["messages"][0] == {
        "role": "tool", "tool_call_id": "toolu_1", "content": '[{"success": true}]',
    }


def test_tools_input_schema_becomes_parameters():
    """工具定义两边字段名不同：input_schema → function.parameters。"""
    adapter = make_adapter(make_response(content="ok"))
    call(adapter, model="test-model", messages=[{"role": "user", "content": "问题"}],
         tools=[{
             "name": "search_knowledge_base",
             "description": "检索知识库",
             "input_schema": {"type": "object",
                              "properties": {"query": {"type": "string"}},
                              "required": ["query"]},
         }])

    tool = adapter._openai.requests[0]["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "search_knowledge_base"
    assert tool["function"]["parameters"]["required"] == ["query"]


def test_tool_choice_conversion():
    """Anthropic 的 {type: ...} → OpenAI 的 auto/required/function 三种写法。"""
    adapter = make_adapter(make_response(content="ok"))

    call(adapter, model="m", messages=[{"role": "user", "content": "x"}],
         tool_choice={"type": "auto"})
    assert adapter._openai.requests[0]["tool_choice"] == "auto"

    call(adapter, model="m", messages=[{"role": "user", "content": "x"}],
         tool_choice={"type": "any"})
    assert adapter._openai.requests[1]["tool_choice"] == "required"

    call(adapter, model="m", messages=[{"role": "user", "content": "x"}],
         tool_choice={"type": "tool", "name": "search_knowledge_base"})
    assert adapter._openai.requests[2]["tool_choice"] == {
        "type": "function", "function": {"name": "search_knowledge_base"},
    }


def test_stop_sequences_become_stop():
    """stop_sequences → stop（字段改名）。"""
    adapter = make_adapter(make_response(content="ok"))
    call(adapter, model="m", stop_sequences=["\n\n"],
         messages=[{"role": "user", "content": "x"}])

    assert adapter._openai.requests[0]["stop"] == ["\n\n"]


# ── 响应方向：OpenAI 响应 → Anthropic 风格 ─────────────────────────────────────

def test_text_response_mapped_back():
    """文本响应翻译回 {type: text} block；stop → end_turn；usage 字段改名。"""
    adapter = make_adapter(make_response(content="建议您多喝水，注意休息"))
    resp = call(adapter, model="m", messages=[{"role": "user", "content": "x"}])

    assert resp.content == [{"type": "text", "text": "建议您多喝水，注意休息"}]
    assert resp.stop_reason == "end_turn"
    assert resp.usage == {"input_tokens": 11, "output_tokens": 7}
    # 关键回归点：翻译后的响应能被现有解析逻辑原样消费
    assert extract_text_content(resp.content) == "建议您多喝水，注意休息"


def test_tool_calls_response_mapped_back():
    """OpenAI tool_calls → {type: tool_use} block；arguments 解析回 dict；stop_reason=tool_use。"""
    adapter = make_adapter(make_response(
        content=None,
        tool_calls=[make_tool_call("search_knowledge_base",
                                   '{"query": "华法林 银杏叶"}')],
        finish_reason="tool_calls",
    ))
    resp = call(adapter, model="m", messages=[{"role": "user", "content": "x"}])

    block = resp.content[0]
    assert block == {
        "type": "tool_use", "id": "toolu_1",
        "name": "search_knowledge_base", "input": {"query": "华法林 银杏叶"},
    }
    assert resp.stop_reason == "tool_use"


def test_text_and_tool_calls_coexist():
    """模型边说话边调工具时：text 和 tool_use 两个 block 都要翻译回来。"""
    adapter = make_adapter(make_response(
        content="让我查一下",
        tool_calls=[make_tool_call("search_knowledge_base", '{"query": "q"}')],
        finish_reason="tool_calls",
    ))
    resp = call(adapter, model="m", messages=[{"role": "user", "content": "x"}])

    assert resp.content[0] == {"type": "text", "text": "让我查一下"}
    assert resp.content[1]["type"] == "tool_use"


def test_invalid_arguments_json_kept_as_raw_string():
    """模型输出的 arguments 不合规时保留原字符串，让上层校验如实报错而不是静默吞掉。"""
    adapter = make_adapter(make_response(
        content=None,
        tool_calls=[make_tool_call("search_knowledge_base", "{不是json")],
        finish_reason="tool_calls",
    ))
    resp = call(adapter, model="m", messages=[{"role": "user", "content": "x"}])

    assert resp.content[0]["input"] == "{不是json"


def test_length_finish_reason_maps_to_max_tokens():
    """finish_reason=length（输出被截断）→ stop_reason=max_tokens。"""
    adapter = make_adapter(make_response(content="没说完", finish_reason="length"))
    resp = call(adapter, model="m", messages=[{"role": "user", "content": "x"}])

    assert resp.stop_reason == "max_tokens"


def test_empty_content_response():
    """content 为 None 且无 tool_calls → 空 block 列表（调用方 extract 后得到空串）。"""
    adapter = make_adapter(make_response(content=None, tool_calls=None))
    resp = call(adapter, model="m", messages=[{"role": "user", "content": "x"}])

    assert resp.content == []
    assert extract_text_content(resp.content) == ""


# ── 工厂函数 ──────────────────────────────────────────────────────────────────

def test_factory_defaults_to_anthropic(monkeypatch):
    """默认（未设置或未知协议值）返回原生 AsyncAnthropic，现有 DeepSeek 链路零变化。"""
    monkeypatch.delenv("OPENMED_LLM_PROTOCOL", raising=False)
    client = create_llm_client(api_key="test", base_url="https://example.com/anthropic")
    assert isinstance(client, AsyncAnthropic)


def test_factory_openai_protocol(monkeypatch):
    """OPENMED_LLM_PROTOCOL=openai 时返回适配器（构造 AsyncOpenAI 不发网络请求）。"""
    monkeypatch.setenv("OPENMED_LLM_PROTOCOL", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")

    client = create_llm_client(api_key="test")

    assert isinstance(client, LLMProtocolAdapter)
    # 工厂把 vLLM 风格的 base_url 和占位 key 正确传给底层 OpenAI 客户端
    # （SDK 内部会把 base_url 规范化成带尾斜杠，对实际请求无影响）
    assert str(client._openai.base_url).rstrip("/") == "http://localhost:8000/v1"


def test_factory_openai_uses_anthropic_base_url_as_fallback(monkeypatch):
    """没配 OPENAI_BASE_URL 时回退用调用方传入的 base_url（保证不留空地址）。"""
    monkeypatch.setenv("OPENMED_LLM_PROTOCOL", "openai")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    client = create_llm_client(api_key="test", base_url="http://localhost:8000/v1")

    assert str(client._openai.base_url).rstrip("/") == "http://localhost:8000/v1"
