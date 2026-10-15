"""LLM 客户端工厂 + 双协议适配器（v2 阶段 0.5）。

背景：全项目统一用 Anthropic 协议调 LLM（DeepSeek 提供了 Anthropic 兼容层，
所以 .env 里 ANTHROPIC_BASE_URL 指过去就能用）。但阶段 4/7 的自训练模型
跑在 vLLM 上，vLLM 只提供 OpenAI 协议（/v1/chat/completions）。

为了让同一套代码既能打 DeepSeek 又能打自训模型，这里做「协议适配」：
    - create_llm_client() 工厂按环境变量 OPENMED_LLM_PROTOCOL 决定返回哪种客户端；
    - LLMProtocolAdapter 对外伪装成 Anthropic 客户端（还是 messages.create(**kwargs)），
      内部把请求翻译成 OpenAI 格式、把响应翻译回 Anthropic 风格 block。

这样 6 个调用点的解析逻辑一行都不用改（_block_type/_block_value 和
extract_text_content 都兼容 dict 形式的 block）。
"""
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from anthropic import AsyncAnthropic


# ── 工厂函数 ──────────────────────────────────────────────────────────────────

def create_llm_client(api_key: str, base_url: Optional[str] = None,
                      protocol: Optional[str] = None):
    """按协议构造 LLM 客户端（protocol 不传时读环境变量 OPENMED_LLM_PROTOCOL）。

    - anthropic（默认）: 返回原生 AsyncAnthropic，DeepSeek 等 Anthropic 兼容服务
      直接可用，与 0.5 之前的行为完全一致；
    - openai: 返回 LLMProtocolAdapter（内部包一个 AsyncOpenAI），用于 vLLM 等
      OpenAI 协议服务。base_url 指向服务的根地址（vLLM 是 http://<ip>:8000/v1），
      api_key 取 OPENAI_API_KEY（vLLM 一般不校验，给占位符 EMPTY 即可）；
      思考开关读 OPENMED_LLM_THINKING（off/on，不设 = 模型默认，见 LLMProtocolAdapter）。

    protocol 显式传入时优先于环境变量：判卷客户端固定传 anthropic（build_judge_cfg），
    被测系统切到 vLLM 时判卷不跟着切（2026-10-15 Qwen3 基线准备）。

    两种返回值的共同点是都有 messages.create(**kwargs)，所以调用方无感知。
    """
    protocol = (protocol or os.getenv("OPENMED_LLM_PROTOCOL", "anthropic")).strip().lower()
    if protocol == "openai":
        # 延迟导入：只有真正切到 OpenAI 协议时才需要这个依赖，
        # 默认的 anthropic 链路不会因为没装 openai 包而启动失败
        from openai import AsyncOpenAI

        return LLMProtocolAdapter(
            AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY", api_key or "EMPTY"),
                base_url=os.getenv("OPENAI_BASE_URL", base_url),
            ),
            thinking=os.getenv("OPENMED_LLM_THINKING"),
        )

    # anthropic 分支：与原代码行为完全一致
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncAnthropic(**kwargs)


# ── 响应壳 ────────────────────────────────────────────────────────────────────

@dataclass
class AnthropicStyleResponse:
    """把 OpenAI 响应翻译回 Anthropic 风格后的返回壳。

    content 是 dict 形式的 block 列表（{"type": "text"/"tool_use", ...}）。
    编排器里的 _block_type/_block_value 和 extract_text_content 都兼容 dict，
    所以调用方拿到这个对象后解析逻辑不用改。
    """
    content: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


# OpenAI 的 finish_reason → Anthropic 的 stop_reason 叫法对照表
_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
}

# Qwen3 思考模式的输出格式：思考过程包在回答开头的 <think>…</think> 里
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _parse_thinking_mode(value: Optional[str]) -> Optional[bool]:
    """OPENMED_LLM_THINKING 的取值 → 思考开关：off→False、on→True、其余（含不设）→None。

    None 表示不替模型做决定（不传开关，按模型默认；Qwen3 默认开思考）。
    """
    mode = (value or "").strip().lower()
    if mode in ("off", "0", "false", "disabled"):
        return False
    if mode in ("on", "1", "true", "enabled"):
        return True
    return None


# ── 适配器 ────────────────────────────────────────────────────────────────────

class LLMProtocolAdapter:
    """OpenAI 协议客户端的外壳，对外提供 Anthropic 风格的 messages.create 接口。

    内部结构示意（调用链）：
        调用方 → adapter.messages.create(**Anthropic风格kwargs)
               → 翻译成 OpenAI 请求体
               → self._openai.chat.completions.create(...)
               → 翻译回 AnthropicStyleResponse
    只实现本项目实际用到的 kwargs 子集：system / messages / tools / tool_choice /
    max_tokens / temperature / stop_sequences / extra_body（思考开关翻译，见下）。

    thinking：思考开关的默认值（工厂从 OPENMED_LLM_THINKING 读），调用方没在
    extra_body 里显式指定时用它。
    """

    def __init__(self, openai_client: Any, thinking: Optional[str] = None):
        self._openai = openai_client
        # 关键伪装：把内部的 chat.completions.create 挂到 messages.create 上，
        # 调用方写 client.messages.create(...) 时走的就是这里
        self.messages = self._Messages(self._openai, _parse_thinking_mode(thinking))

    class _Messages:
        def __init__(self, client: Any, thinking_default: Optional[bool] = None):
            self._client = client
            self._thinking_default = thinking_default

        async def create(self, **kwargs: Any) -> AnthropicStyleResponse:
            request = self._to_openai_request(kwargs, self._thinking_default)
            resp = await self._client.chat.completions.create(**request)
            return self._to_anthropic_response(resp)

        # ── 请求方向：Anthropic kwargs → OpenAI 请求体 ────────────────────────

        @staticmethod
        def _to_openai_request(kwargs: Dict[str, Any],
                               thinking_default: Optional[bool] = None) -> Dict[str, Any]:
            request: Dict[str, Any] = {}

            # model / temperature / max_tokens 两边同名，直接透传
            for key in ("model", "temperature", "max_tokens"):
                if key in kwargs:
                    request[key] = kwargs[key]

            # OpenAI 没有独立的 system 参数：把 system 拼成消息列表头部的
            # system 角色消息（Anthropic 允许 system 是 block 列表，这里只取文本块）
            messages: List[Dict[str, Any]] = []
            system = kwargs.get("system")
            if system:
                if isinstance(system, str):
                    system_text = system
                else:
                    system_text = "\n".join(
                        block.get("text", "") for block in system
                        if block.get("type") == "text"
                    )
                messages.append({"role": "system", "content": system_text})

            for message in kwargs.get("messages", []):
                messages.extend(_convert_message(message))
            request["messages"] = messages

            # tools：Anthropic 的 input_schema → OpenAI 的 function.parameters
            tools = kwargs.get("tools")
            if tools:
                request["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("input_schema", {"type": "object"}),
                        },
                    }
                    for tool in tools
                ]

            # tool_choice：Anthropic 是 {type: ...} 结构，OpenAI 是字符串或嵌套 dict
            tool_choice = kwargs.get("tool_choice")
            if tool_choice:
                request["tool_choice"] = _convert_tool_choice(tool_choice)

            # stop_sequences → stop
            if "stop_sequences" in kwargs:
                request["stop"] = kwargs["stop_sequences"]

            # 思考开关（2026-10-15 Qwen3 基线准备）：两边写法不同——
            #   Anthropic/DeepSeek：extra_body={"thinking": {"type": "disabled"}}
            #   Qwen3（vLLM）：     extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            #   （Qwen3 的开关做在 chat template 里，vLLM 把这个字段转交给模板）
            # 优先级：调用方显式传的 thinking > 适配器默认值（OPENMED_LLM_THINKING）> 不传。
            # 现有调用点（记忆摘要/画像/合并、判卷）显式关思考，翻译后对 Qwen3 同样生效；
            # 主链路/意图识别没传，靠 OPENMED_LLM_THINKING=off 统一关。
            # extra_body 里其他键原样转发（OpenAI SDK 会并进请求体）。
            extra_body = dict(kwargs.get("extra_body") or {})
            thinking = extra_body.pop("thinking", None)
            enable_thinking = thinking_default
            if isinstance(thinking, dict) and thinking.get("type") in ("disabled", "enabled"):
                enable_thinking = thinking["type"] == "enabled"
            if enable_thinking is not None:
                template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
                template_kwargs.setdefault("enable_thinking", enable_thinking)
                extra_body["chat_template_kwargs"] = template_kwargs
            if extra_body:
                request["extra_body"] = extra_body

            return request

        # ── 响应方向：OpenAI 响应 → Anthropic 风格 ────────────────────────────

        @staticmethod
        def _to_anthropic_response(resp: Any) -> AnthropicStyleResponse:
            """把 OpenAI 响应翻译回 Anthropic 风格。

            OpenAI 的 content 是字符串、工具调用在独立的 tool_calls 数组里；
            Anthropic 的 content 是 block 列表（text 和 tool_use 混在一起）。
            这里重新拼回 block 列表。
            """
            choice = resp.choices[0]
            message = choice.message

            blocks: List[Dict[str, Any]] = []
            # 文本部分：OpenAI 的 content 可能为 None（纯工具调用时）
            text = getattr(message, "content", None)
            if isinstance(text, str):
                # 防御：思考开关没生效时，Qwen3 会把思考过程写在回答开头的
                # <think>…</think> 里——去掉，不让思考内容混进最终回答和轨迹
                # （vLLM 开了 reasoning parser 时思考走单独字段，这里本来就看不到）
                text = _THINK_BLOCK.sub("", text, count=1)
            if isinstance(text, str) and text:
                blocks.append({"type": "text", "text": text})

            # 工具调用部分：arguments 是 JSON 字符串，翻译回 dict；
            # 解析失败（模型输出不合规）时保留原始字符串，让上层校验环节如实报错
            for tool_call in getattr(message, "tool_calls", None) or []:
                raw_args = tool_call.function.arguments or "{}"
                try:
                    tool_input = json.loads(raw_args)
                except json.JSONDecodeError:
                    tool_input = raw_args
                blocks.append({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": tool_input,
                })

            usage = None
            if getattr(resp, "usage", None) is not None:
                usage = {
                    "input_tokens": resp.usage.prompt_tokens,
                    "output_tokens": resp.usage.completion_tokens,
                }

            return AnthropicStyleResponse(
                content=blocks,
                stop_reason=_FINISH_REASON_MAP.get(choice.finish_reason),
                usage=usage,
            )


# ── 消息级转换（模块级函数，方便单测直接调用）──────────────────────────────────

def _convert_message(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把一条 Anthropic 消息翻译成一条或多条 OpenAI 消息。

    三种情况：
    1. content 是纯字符串 → 原样一条消息（最常见）；
    2. assistant 消息带 tool_use block → 拆出 tool_calls 数组，
       input 要序列化成 JSON 字符串（OpenAI 的 arguments 是字符串）；
    3. user 消息带 tool_result block → 每条结果变成一条 role="tool" 的独立消息
       （OpenAI 把工具结果当成独立角色，而 Anthropic 把它塞在 user 消息里）。
    """
    content = message.get("content")
    role = message.get("role", "user")

    # 情况 1：纯文本
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        return [{"role": role, "content": str(content or "")}]

    # 把 block 列表按类型分组
    text_parts: List[str] = []
    tool_uses: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    for block in content:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            tool_uses.append(block)
        elif block_type == "tool_result":
            tool_results.append(block)

    converted: List[Dict[str, Any]] = []

    # 情况 2：assistant 的工具调用。text 和 tool_calls 可以同一条消息共存
    if tool_uses:
        assistant_msg: Dict[str, Any] = {"role": "assistant"}
        if text_parts:
            assistant_msg["content"] = "\n".join(text_parts)
        assistant_msg["tool_calls"] = [
            {
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    # ensure_ascii=False：中文参数原样保留，不转成 \uXXXX
                    "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                },
            }
            for block in tool_uses
        ]
        converted.append(assistant_msg)
    elif text_parts:
        converted.append({"role": role, "content": "\n".join(text_parts)})

    # 情况 3：工具结果 → role="tool" 的独立消息。
    # 必须紧跟在前面的 assistant 消息之后（OpenAI 会校验 tool_call_id 找得到出处）
    for block in tool_results:
        converted.append({
            "role": "tool",
            "tool_call_id": block.get("tool_use_id", ""),
            "content": block.get("content", ""),
        })

    return converted


def _convert_tool_choice(tool_choice: Dict[str, Any]) -> Any:
    """Anthropic 的 tool_choice 结构 → OpenAI 的三种写法。

    Anthropic: {"type": "auto"} / {"type": "any"} / {"type": "tool", "name": "xxx"}
    OpenAI:    "auto"           / "required"        / {"type": "function", "function": {"name": "xxx"}}
    """
    choice_type = tool_choice.get("type", "auto")
    if choice_type == "auto":
        return "auto"
    if choice_type in ("any", "required"):
        return "required"
    if choice_type == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    if choice_type == "none":
        return "none"
    return "auto"


# ── 通用解析辅助（原有功能）───────────────────────────────────────────────────

def extract_text_content(content: Iterable[Any]) -> str:
    """Return text blocks from Anthropic-style response content."""
    texts: List[str] = []
    for block in content or []:
        if isinstance(block, str):
            texts.append(block)
            continue

        block_type = getattr(block, "type", None)
        text = getattr(block, "text", None)
        if isinstance(block, dict):
            block_type = block.get("type", block_type)
            text = block.get("text", text)

        if isinstance(text, str) and (block_type in (None, "text")):
            texts.append(text)

    return "\n".join(t for t in texts if t)
