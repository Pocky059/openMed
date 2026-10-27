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
