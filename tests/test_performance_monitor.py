"""性能监控 Prometheus 指标注册测试（2026-10-26 接通业务指标输出）。

教训背景：指标注册原来绑在 start_http_server 一起、被 PROMETHEUS_PORT
默认 0 挡死——/metrics 端点里只有 python 平台指标，业务指标从未注册，
Prometheus 容器抓的是空仓库。注册现在永远执行（零成本、不占端口），
独立端口（start_http_server）才是可选项。
"""
from prometheus_client import REGISTRY

from monitor.performance_monitor import PerformanceMonitor


def _make_monitor(**kwargs):
    """构造只用于注册测试的监控器：orchestrator/tool_manager 构造时只存
    引用（_collect 才调用 get_stats），假对象即可。"""
    return PerformanceMonitor(
        orchestrator=object(),
        tool_manager=object(),
        interval_s=10.0,
        **kwargs,
    )


def test_metrics_registered_without_port():
    """不设 prometheus_port 也注册业务指标——/metrics 端点有得输出。"""
    m = _make_monitor(prometheus_port=None)
    assert "agent_success_rate" in m._prom
    assert "agent_latency_ms" in m._prom
    assert "tool_success_rate" in m._prom
    assert "requests_total" not in m._prom  # 死指标已删：没有埋点不注册


def test_metrics_visible_in_default_registry():
    """业务指标进默认 REGISTRY——generate_latest()（/metrics 端点）能输出。"""
    _make_monitor(prometheus_port=None)
    names = [metric.name for metric in REGISTRY.collect()]
    assert "agent_success_rate" in names


def test_duplicate_construction_does_not_raise():
    """重复构造不炸（模块级单例复用）：prometheus_client 同名指标重复注册抛 ValueError。"""
    _make_monitor(prometheus_port=None)
    _make_monitor(prometheus_port=None)  # 不抛就是通过


# ── 路由惩罚（2026-10-27 阈值修正：DeepSeek 天然延迟不再误伤路由评分） ──────

def test_routing_penalty_no_penalty_for_natural_llm_latency():
    """34s（DeepSeek 全链路实测典型值）< 60s 新起点 => 零延迟惩罚。
    曾设 15000ms 起点：每次正常调用都吃 0.4 惩罚、告警刷屏（固定阈值误报）。"""
    assert PerformanceMonitor._routing_penalty(1.0, 34000) == 0.0
    assert PerformanceMonitor._routing_penalty(1.0, 15000) == 0.0


def test_routing_penalty_ramps_after_new_start():
    """超过 60s 才起罚：72s（超 1/5）罚 0.2，120s（超 1 倍）封顶 0.4。"""
    assert PerformanceMonitor._routing_penalty(1.0, 72000) == 0.2
    assert PerformanceMonitor._routing_penalty(1.0, 120000) == 0.4


def test_routing_penalty_success_rate_part_unchanged():
    """成功率惩罚不受阈值修正影响：85% => 0.1（浮点误差范围内）。"""
    assert abs(PerformanceMonitor._routing_penalty(0.85, 1000) - 0.1) < 1e-9
