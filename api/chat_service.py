"""
对话服务层（v2 阶段 1 第 1 步：重构地基）。

把 /chat 端点的五步主链路从 api/main.py 抽出来：
  记忆读取 → 意图识别 → Agent 路由 → 执行 → 记忆写入 → 画像更新

抽出的理由：HTTP 层只做绑定与序列化，业务五步属于服务层——单测可以直接
构造 ChatService 用假组件测五步，不必起 FastAPI；将来若有第二入口（如
基线评测 CLI 的对话回放），复用逻辑而非复制。
"""
import asyncio
import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ChatResponse(BaseModel):         #这里就是回答体了。他会自动把回答转成JSON输出。
    conv_id:     str                   #必填字段：对话id
    request_id:  str = ""
    response:    str                   #必填字段：模型回答
    intent:      str                   #必填字段：识别出的意图
    intent_group: str = "other"
    agent_type:  str                   #必填字段：处理该请求的agent类型
    agent_types: List[str] = Field(default_factory=list)         #使用可变默认值写法
    primary_agent: str = ""
    supporting_agents: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    escalated:   bool                  #必填字段：是否转人工
    urgency:     str = "LOW"           #紧急级别（UrgencyLevel 枚举值，如 CRITICAL；急症集判分对比期望用）
    latency_ms:  float                 #必填字段：处理耗时（毫秒）
    knowledge_used: bool = False
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    intent_confidence: float = 0.0
    intent_source_scores: Dict[str, float] = Field(default_factory=dict)


class ChatService:
    """五步主链路的服务封装：只持有编排器与记忆管理器的引用，无自身状态。"""

    def __init__(self, orchestrator, memory):
        self._orchestrator = orchestrator   #Agent路由与编排器
        self._memory       = memory         #上下文记忆管理器

    async def handle_chat_turn(
        self,
        message: str,
        user_id: str,
        conv_id: Optional[str] = None,
    ) -> ChatResponse:
        """处理一轮对话。完整流程：
          记忆读取 → 意图识别 → Agent 路由 → 执行 → 记忆写入 → 画像更新
        """
        from agents.agent_orchestrator import Request as OrcReq
        from memory.conversation_memory import MsgRole

        conv_id = conv_id or str(uuid.uuid4())            #uuid：python库，用于生成唯一标识符，基于随机数

        # 1. 读取记忆上下文（工作记忆+情景记忆+用户画像一起保存成的上下文；异步执行。注意这里的mem_ctx和后面的full_context都不是ui上显示的文本）
        mem_ctx = await self._memory.get_context(user_id, conv_id, query=message)

        # 2. 构建编排请求（含对话历史，用于意图识别上下文）
        history = [
            {"role": m.role.value, "content": m.content} #从mem_ctx中抽取每个m形成字典列表，即格式化上下文
            for m in mem_ctx.recent_messages[-5:]        #只取最近的五条消息，因此主要是redis工作记忆
        ] if mem_ctx.recent_messages else None

        intent_result = await self._orchestrator.recognize_intent(message, history=history)   #把请求体里的message（当前信息）和工作记忆作为输入发送到意图识别模块，得到返回
        full_context = mem_ctx.to_prompt_text()   #将整个mem_ctx记忆上下文格式化为大模型可用的prompt，起整体参考作用

        orch_req = OrcReq( #把结果先包装成一个请求体，再喂给编排器。这里的请求体OrcReq是Request类，它是在orchestrator里被定义的。
            message=message,
            user_id=user_id,
            conv_id=conv_id,           #这两个记忆现在都被送到Request中，计算agent路由了：
            context=full_context,      #并入历史（格式化综合上下文记忆）
            history=history,           #工作记忆（结构化）
            entities=intent_result.entities,
            intent=intent_result.intent,
            intent_group=intent_result.intent_group,
            urgency=intent_result.urgency,
            intent_confidence=intent_result.confidence,
        )

        # 3. 送入编排器，决定接下来交给哪个agent，用什么工具，要不要多agent等
        result = await self._orchestrator.run(orch_req)

        # 4. 写入记忆（写入redis，工作记忆）
        await self._memory.add_message(user_id, conv_id, MsgRole.USER, message)   #写入用户提问
        await self._memory.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)    #写入模型回答

        # 5. 异步更新用户画像（不阻塞响应）
        asyncio.create_task(self._memory.update_profile(user_id, conv_id))

        return ChatResponse(    #前面定义的返回体，包含若干参数。
            conv_id=conv_id,
            request_id=result.request_id,
            response=result.response,
            intent=result.intent.value if result.intent else "other",
            intent_group=intent_result.intent_group,
            agent_type=result.agent_type.value,
            agent_types=[agent_type.value for agent_type in result.agent_types],    #result里有两个字段，一个agent_type, 一个agent_types
            primary_agent=result.primary_agent.value if result.primary_agent else result.agent_type.value,  #此外还有一个primary_agent字段。没有primary就用agent_type
            supporting_agents=[agent_type.value for agent_type in result.supporting_agents],     #还有一个supporting_agents字段
            tools_used=result.tools_used,
            routing_reason=result.routing_reason,
            routing_confidence=result.routing_confidence,
            escalated=result.escalated,
            urgency=intent_result.urgency.name if intent_result.urgency else "LOW",
            latency_ms=round(result.latency_ms, 1),
            knowledge_used="search_knowledge_base" in result.tools_used,
            entities=intent_result.entities,
            intent_confidence=round(intent_result.confidence, 4),
            intent_source_scores=intent_result.source_scores,
        )
