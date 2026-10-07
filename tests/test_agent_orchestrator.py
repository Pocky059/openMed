import asyncio

from agents.agent_orchestrator import (
    AgentProfile,
    AgentResponse,
    AgentType,
    AgentOrchestrator,
    AppointmentAgent,
    EmergencyAgent,
    MedicationAgent,
    Request,
    ResponseComposer,
    RoutingDecision,
    SymptomTriageAgent,
    build_shared_rag_tools,
)
from agents.tools import AgentToolSpec
from core.intent_recognizer import IntentCategory, UrgencyLevel


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

        class Messages:
            async def create(inner, **kwargs):
                self.calls.append(kwargs)
                if self.error:
                    raise self.error
                return self.response

        self.messages = Messages()


def make_request(**kwargs):
    values = {
        "message": "我想问用药的剂量，同时想预约复诊",
        "user_id": "u1",
        "conv_id": "c1",
        "intent": IntentCategory.MEDICATION_DOSAGE,
        "intent_group": "medication",
        "urgency": UrgencyLevel.HIGH,
        "intent_confidence": 0.92,
        "entities": {"drug_name": ["布洛芬"], "appointment_id": ["A20260501"]},
    }
    values.update(kwargs)
    return Request(**values)


def test_agent_profiles_have_distinct_contracts_and_generation_config():
    assert isinstance(SymptomTriageAgent.profile, AgentProfile)
    assert SymptomTriageAgent.profile.role != MedicationAgent.profile.role
    assert MedicationAgent.profile.workflow != AppointmentAgent.profile.workflow
    assert MedicationAgent.profile.temperature < SymptomTriageAgent.profile.temperature
    assert "search_knowledge_base" in SymptomTriageAgent.profile.tool_scope
    assert "build_medication_plan" in MedicationAgent.profile.tool_scope
    assert "check_appointment_fields" in AppointmentAgent.profile.tool_scope


def test_domain_agents_build_different_role_packets():
    req = make_request()
    symptom_packet = SymptomTriageAgent(FakeClient(), "test-model")._build_role_packet(req)
    medication_packet = MedicationAgent(FakeClient(), "test-model")._build_role_packet(req)
    appointment_packet = AppointmentAgent(FakeClient(), "test-model")._build_role_packet(req)

    assert "triage_targets" in symptom_packet
    assert "medication_fields" in medication_packet
    assert "verification_fields" in appointment_packet
    assert symptom_packet != medication_packet != appointment_packet


def test_emergency_agent_is_a_real_non_llm_handoff_node():
    client = FakeClient()
    agent = EmergencyAgent(client, "test-model")

    result = asyncio.run(agent.handle(make_request(
        intent=IntentCategory.HUMAN_HANDOFF,
        urgency=UrgencyLevel.CRITICAL,
    )))

    assert result.success is True
    assert result.escalate is True
    assert "急症" in result.content and "交接" in result.content
    assert client.calls == []


def test_composer_fallback_preserves_primary_and_supporting_results():
    composer = ResponseComposer(FakeClient(error=RuntimeError("provider down")), "test-model")
    req = make_request()
    responses = [
        AgentResponse(AgentType.MEDICATION, "先确认布洛芬的用法用量。", True),
        AgentResponse(AgentType.APPOINTMENT, "请提供预约单号和期望的新时间。", True),
    ]

    content = asyncio.run(composer.compose(req, responses))

    assert content.startswith("先确认布洛芬的用法用量。")
    assert "补充说明" in content
    assert "预约单号" in content


def test_routing_decision_can_target_emergency_pool():
    # Keep this assertion close to the public data contract used by the API.
    decision = RoutingDecision(
        primary_agent=AgentType.EMERGENCY,
        reason="critical request",
        confidence=1.0,
    )
    assert decision.agent_types == [AgentType.EMERGENCY]
    assert not decision.multi_agent


def test_composite_request_routes_explicit_appointment_signal_as_supporting_agent():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.SYMPTOM_TRIAGE: [object()],
        AgentType.MEDICATION: [object()],
        AgentType.APPOINTMENT: [object()],
    }

    decision = orchestrator._route_decision(make_request())

    assert decision.primary_agent is AgentType.MEDICATION
    assert decision.supporting_agents == [AgentType.APPOINTMENT]
    assert decision.multi_agent is True


def test_agent_tool_scopes_are_real_and_isolated():
    symptom_tools = set(SymptomTriageAgent(FakeClient(), "test-model").get_tools())
    medication_tools = set(MedicationAgent(FakeClient(), "test-model").get_tools())
    appointment_tools = set(AppointmentAgent(FakeClient(), "test-model").get_tools())
    emergency_tools = set(EmergencyAgent(FakeClient(), "test-model").get_tools())

    assert symptom_tools == {"inspect_request_context", "suggest_required_fields"}
    assert medication_tools == {"build_medication_plan"}
    assert appointment_tools == {"check_appointment_fields", "compare_registration_fees"}
    # 急症响应是硬编码文本（不走 LLM/工具，保证确定性与零延迟），
    # 白名单必须如实为空——避免"声称可用、实际永不执行"的死工具
    assert emergency_tools == set()
    assert not symptom_tools & medication_tools
    assert not medication_tools & appointment_tools


def test_shared_rag_tool_is_available_to_all_agents():
    class RagManager:
        async def search_with_rerank(self, tool_name, query, top_k=5):
            return type(
                "Result",
                (),
                {"success": True, "data": [{"title": "预约挂号流程说明", "content": "预约成功后会生成预约单号"}], "reranked": True},
            )()

    shared = build_shared_rag_tools(RagManager())

    symptom = SymptomTriageAgent(FakeClient(), "test-model")
    medication = MedicationAgent(FakeClient(), "test-model")
    appointment = AppointmentAgent(FakeClient(), "test-model")
    emergency = EmergencyAgent(FakeClient(), "test-model")

    for agent in (symptom, medication, appointment, emergency):
        agent.set_shared_tools(shared)
        tools = agent.get_tools()
        assert "search_knowledge_base" in tools


def test_tool_input_validation_rejects_unknown_fields():
    agent = MedicationAgent(FakeClient(), "test-model")
    spec = agent.get_tools()["build_medication_plan"]

    try:
        agent._validate_tool_input(spec, {"current_medications": "华法林", "has_allergy": False, "secret": "nope"})
    except ValueError as exc:
        assert "不允许的工具参数" in str(exc)
    else:
        raise AssertionError("unknown tool fields should be rejected")


def test_tool_use_round_trip_executes_only_whitelisted_tool():
    class ToolUseBlock:
        type = "tool_use"
        id = "toolu_1"
        name = "build_medication_plan"
        input = {"current_medications": "华法林", "has_allergy": False}

    class TextBlock:
        type = "text"
        text = "已根据现用药清单生成用药核查步骤。"

    class ToolClient:
        def __init__(self):
            self.calls = []
            self.responses = [
                type("Response", (), {"content": [ToolUseBlock()]})(),
                type("Response", (), {"content": [TextBlock()]})(),
            ]

        class Messages:
            def __init__(self, owner):
                self.owner = owner

            async def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return self.owner.responses.pop(0)

        @property
        def messages(self):
            return self.Messages(self)

    client = ToolClient()
    agent = MedicationAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert response.tools_used == ["build_medication_plan"]
    assert len(client.calls) == 2
    assert {tool["name"] for tool in client.calls[0]["tools"]} == {
        "build_medication_plan",
    }
    assert "tool_result" in str(client.calls[1]["messages"])


def test_success_path_keeps_final_answer_and_tool_traces():
    """回归（审查 H1/H3）：正常结束的请求，消息流必须以最终回答结尾、
    tool_traces 必须保留。此前提前 return 分支漏设 _last_tool_traces（/trace 接口
    恒为空）且不 append 最终回答（轨迹 SFT 模具缺 final）。"""

    class ToolUseBlock:
        type = "tool_use"
        id = "toolu_1"
        name = "build_medication_plan"
        input = {"current_medications": "华法林", "has_allergy": False}

    class TextBlock:
        type = "text"
        text = "已根据现用药清单生成用药核查步骤。"

    class ToolClient:
        def __init__(self):
            self.calls = []
            self.responses = [
                type("Response", (), {"content": [ToolUseBlock()]})(),
                type("Response", (), {"content": [TextBlock()]})(),
            ]

        class Messages:
            def __init__(self, owner):
                self.owner = owner

            async def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return self.owner.responses.pop(0)

        @property
        def messages(self):
            return self.Messages(self)

    client = ToolClient()
    agent = MedicationAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request()))

    # H3：正常结束（含工具调用轮）的请求 tool_traces 不能被 handle 开头的 reset 清空
    assert response.tool_traces, "正常结束的请求 tool_traces 不应为空"
    assert response.tool_traces[0]["tool_name"] == "build_medication_plan"
    # H1：消息流以最终回答（assistant 纯文本）结尾，SFT 模具才完整
    assert response.messages[-1]["role"] == "assistant"
    assert response.messages[-1]["content"] == "已根据现用药清单生成用药核查步骤。"
