"""Agent 工具定义与实现。

所有 Agent 工具集中在这里，编排器只负责：
  1. 根据 Agent 类型暴露工具白名单
  2. 执行 LLM 返回的 tool_use
  3. 将工具结果回传给 LLM

工具本身保持确定性、可测试，并明确区分：
  - 当前请求分析
  - 用药信息与用药安全提示
  - 预约字段核验
  - 急症/人工交接摘要
  - 共享知识库 RAG

预约操作、处方开具、费用退款等需要真实医院系统授权的动作不在这里伪造。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from agents.agent_orchestrator import Request


AgentToolHandler = Callable[["Request", Dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class AgentToolSpec:
    """Agent 可见工具的定义和执行函数。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: AgentToolHandler


def make_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    handler: AgentToolHandler,
    required: Optional[List[str]] = None,
) -> AgentToolSpec:
    """创建带 JSON Schema 的 Agent 工具。"""
    return AgentToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler,
    )


def inspect_request_context(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """分诊工具：返回脱敏后的当前请求快照。"""
    return {
        "intent": req.intent.value if req.intent else None,
        "intent_group": req.intent_group,
        "urgency": req.urgency.name if req.urgency else None,
        "intent_confidence": round(req.intent_confidence, 4),
        "entities": req.entities or {},
        "context_available": bool(req.context),
        "requested_focus": str(args.get("focus", "general"))[:40],
    }


def suggest_required_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """分诊工具：按业务类型计算下一轮只需询问的字段。"""
    intent = req.intent.value if req.intent else "other"
    fields: List[str] = []
    if intent in {"symptom_check", "department_guide"}:
        fields = ["症状出现时间", "症状严重程度或是否持续加重"]
    elif intent in {"appointment_manage", "appointment_reschedule"}:
        fields = ["预约单号或原就诊时间", "期望的新时间或科室"]
    elif intent in {"complaint", "request"}:
        fields = ["事件发生时间", "期望的处理方式"]
    elif intent == "other":
        fields = ["希望咨询的具体症状或问题"]
    return {
        "intent": intent,
        "required_fields": fields,
        "known_entities": req.entities or {},
    }


_DRUG_KB: Dict[str, tuple[str, List[str]]] = {
    "布洛芬": ("解热镇痛药，常用于发热、头痛、关节痛等", [
        "避免空腹服用，胃部不适者建议饭后服用",
        "不与其他非甾体抗炎药同时使用",
        "孕妇及严重肝肾功能不全者慎用",
    ]),
    "对乙酰氨基酚": ("解热镇痛药，常用于退烧、轻中度疼痛", [
        "严格按说明书剂量服用，避免与其他含相同成分药物叠加",
        "肝功能异常者慎用",
        "24 小时内服用不超过说明书标注上限",
    ]),
    "阿莫西林": ("青霉素类抗生素，用于细菌感染", [
        "需遵医嘱按疗程服用，不可自行停药",
        "青霉素过敏者禁用",
        "服药期间避免饮酒",
    ]),
    "头孢类抗生素": ("头孢菌素类抗生素，用于细菌感染", [
        "服药前后避免饮酒，存在双硫仑样反应风险",
        "有青霉素/头孢过敏史者需告知医生",
    ]),
}


def lookup_drug_info(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """用药工具：给出常见药品的保守用药提示，不声称读取了药品说明书原文。"""
    name = str(args.get("drug_name", "")).strip()
    meaning, steps = _DRUG_KB.get(
        name,
        ("暂未收录该药品的结构化信息", ["建议查看药品说明书或咨询药师", "如有基础疾病或正在服用其他药物，请告知医生或药师"]),
    )
    return {
        "drug_name": name,
        "usage_summary": meaning,
        "guidance": steps,
        "package_insert_checked": False,
    }


def build_medication_plan(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """用药工具：结合现用药和过敏史生成保守的用药核查步骤。"""
    current_medications = str(args.get("current_medications", "无"))[:200]
    has_allergy = bool(args.get("has_allergy", False))
    steps = [
        "核对药品名称、剂型和有效期",
        "确认是否与现用药物存在已知相互作用",
        "按说明书或医嘱确认单次剂量和服用频次",
    ]
    if has_allergy:
        steps.append("已知过敏史，建议服药前再次核实成分并咨询药师")
    return {
        "current_medications": current_medications,
        "has_allergy": has_allergy,
        "guidance_steps": steps,
    }


def check_appointment_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """预约工具：检查必要核验字段是否齐全。"""
    fields = {
        "appointment_id": bool(req.entities.get("appointment_id")),
        "department": bool(req.entities.get("department")),
        "date": bool(req.entities.get("date")),
        "appointment_type": bool(args.get("appointment_type")),
    }
    return {
        "fields": fields,
        "missing_fields": [name for name, present in fields.items() if not present],
        "can_confirm_change": False,
        "reason": "当前工具只做字段检查，不连接挂号或支付系统",
    }


def compare_registration_fees(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """预约工具：只做用户明确提供挂号费之间的算术。"""
    try:
        first = float(args["fee_a"])
        second = float(args["fee_b"])
    except (KeyError, TypeError, ValueError):
        return {"success": False, "error": "fee_a 和 fee_b 必须是数字"}
    return {
        "success": True,
        "fee_a": first,
        "fee_b": second,
        "difference": round(first - second, 2),
        "interpretation": "仅表示挂号费差值，不代表重复扣款或退费结论",
    }


def create_handoff_summary(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """急症/升级工具：生成可交给人工或急诊的结构化摘要。"""
    return {
        "request_id": req.request_id,
        "reason": str(args.get("reason", "需要人工或急诊进一步处理"))[:120],
        "intent": req.intent.value if req.intent else "unknown",
        "urgency": req.urgency.name if req.urgency else "UNKNOWN",
        "entities": req.entities or {},
        "sensitive_data_required": False,
    }


def build_shared_rag_tools(tool_manager: Any) -> Dict[str, AgentToolSpec]:
    """构建所有 Agent 可共享的 RAG 工具。"""

    async def search_knowledge_base(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or req.message or "").strip()
        top_k = int(args.get("top_k", 5) or 5)
        if not query:
            return {"success": False, "error": "query 不能为空", "results": []}
        if tool_manager is None:
            return {"success": False, "error": "RAG 工具未初始化", "results": []}

        result = await tool_manager.search_with_rerank(
            "knowledge_search",
            query,
            top_k=top_k,
        )
        if not getattr(result, "success", False):
            return {
                "success": False,
                "query": query,
                "error": getattr(result, "error", "知识库检索失败"),
                "results": [],
                "reranked": False,
            }

        return {
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": result.data,
            "reranked": bool(getattr(result, "reranked", False)),
        }

    return {
        "search_knowledge_base": make_tool(
            "search_knowledge_base",
            "检索医疗知识库并返回最相关的文档片段；可用于症状分诊、用药咨询、预约挂号和急症升级场景。",
            {
                "query": {"type": "string", "description": "用户问题或检索关键词"},
                "top_k": {"type": "integer", "description": "返回结果条数"},
            },
            search_knowledge_base,
            required=["query"],
        )
    }


def symptom_triage_tools() -> Dict[str, AgentToolSpec]:
    return {
        "inspect_request_context": make_tool(
            "inspect_request_context",
            "查看当前请求的意图、紧急度、实体和上下文可用性；不查询外部业务系统。",
            {"focus": {"type": "string", "description": "希望关注的业务方向"}},
            inspect_request_context,
        ),
        "suggest_required_fields": make_tool(
            "suggest_required_fields",
            "根据当前意图建议下一轮只需向用户补充的字段。",
            {},
            suggest_required_fields,
        ),
    }


def medication_tools() -> Dict[str, AgentToolSpec]:
    return {
        "lookup_drug_info": make_tool(
            "lookup_drug_info",
            "查询常见药品的用途和保守用药提示；不会替代药品说明书或医嘱。",
            {"drug_name": {"type": "string", "description": "药品名称，例如布洛芬"}},
            lookup_drug_info,
            required=["drug_name"],
        ),
        "build_medication_plan": make_tool(
            "build_medication_plan",
            "根据现用药清单和过敏史生成用药安全核查步骤，不执行处方调整。",
            {
                "current_medications": {"type": "string", "description": "当前正在服用的其他药物，逗号分隔"},
                "has_allergy": {"type": "boolean", "description": "是否已知有药物过敏史"},
            },
            build_medication_plan,
            required=["current_medications", "has_allergy"],
        ),
    }


def appointment_tools() -> Dict[str, AgentToolSpec]:
    return {
        "check_appointment_fields": make_tool(
            "check_appointment_fields",
            "检查预约核验字段是否齐全；不连接挂号或支付系统。",
            {"appointment_type": {"type": "string", "description": "挂号类型，例如普通号、专家号"}},
            check_appointment_fields,
        ),
        "compare_registration_fees": make_tool(
            "compare_registration_fees",
            "计算用户明确提供的两笔挂号费差值；不判断是否重复扣款，也不执行退费。",
            {
                "fee_a": {"type": "number", "description": "第一笔挂号费"},
                "fee_b": {"type": "number", "description": "第二笔挂号费"},
            },
            compare_registration_fees,
            required=["fee_a", "fee_b"],
        ),
    }


def emergency_tools() -> Dict[str, AgentToolSpec]:
    return {
        "create_handoff_summary": make_tool(
            "create_handoff_summary",
            "生成交给人工/急诊的结构化交接摘要，可用于对接医院工单系统；不会创建真实工单。",
            {"reason": {"type": "string", "description": "需要升级的原因"}},
            create_handoff_summary,
        ),
    }
