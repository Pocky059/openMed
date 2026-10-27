"""Skill 上传与内置保护测试（2026-10-27 前端 Agent 管理页）。

覆盖 core/skill_loader.save_uploaded 与 /skills/upload 端点：
上传落独立目录、内置 Skill 同名拒绝、坏文件清理、格式校验。
"""
import asyncio

import pytest

from core.skill_loader import SkillManager


@pytest.fixture
def manager(tmp_path):
    """内置目录放一个 SKILL.md，上传目录独立，模拟生产 skills/ + skills_uploads/。"""
    root = tmp_path / "skills"
    uploads = tmp_path / "skills_uploads"
    builtin_dir = root / "refund"
    builtin_dir.mkdir(parents=True)
    (builtin_dir / "SKILL.md").write_text(
        "---\nname: 退款处理\nkeywords: 退款, 退费\ndescription: 内置退款流程\n---\n"
        "用户申请退款时，先核验支付凭证，再走财务审核。\n",
        encoding="utf-8",
    )
    manager = SkillManager(root_dir=str(root), uploads_dir=str(uploads))
    manager.load()  # 生产组装（build_skill_manager）里显式 load，测试同样
    return manager


def test_save_uploaded_marks_source_and_reloads(manager):
    """上传的 Skill 落上传目录、热加载生效、source=uploaded。"""
    skill = manager.save_uploaded(
        "投诉安抚.md",
        "---\nname: 投诉安抚\nkeywords: 投诉, 不满意\ndescription: 安抚话术\n---\n"
        "先道歉，再记录诉求，承诺 24 小时内回复。\n",
    )
    assert skill.name == "投诉安抚"
    assert skill.source == "uploaded"
    # 已热加载：再次 load 后仍在列表里（文件真实落盘）
    names = {s.name: s for s in manager.load()}
    assert "投诉安抚" in names
    assert names["投诉安抚"].source == "uploaded"
    assert names["退款处理"].source == "builtin"


def test_save_uploaded_json_format(manager):
    """JSON 格式上传（与内置 JSON Skill 解析规则一致）。"""
    skill = manager.save_uploaded(
        "greeting.json",
        '{"name": "问候语", "content": "问候时使用敬语。", "keywords": ["你好"], "agents": ["appointment"]}',
    )
    assert skill.name == "问候语"
    assert skill.keywords == ["你好"]
    assert skill.agents == ["appointment"]
    assert skill.source == "uploaded"


def test_save_uploaded_rejects_builtin_name_collision(manager):
    """与内置 Skill 同名 => 拒绝并清理文件，内置文件原样保留。"""
    with pytest.raises(ValueError, match="内置 Skill 不可改动"):
        manager.save_uploaded("其他.md", "---\nname: 退款处理\n---\n假装覆盖内置。\n")
    # 上传目录被清理干净，内置仍在
    names = [s.name for s in manager.load()]
    assert names == ["退款处理"]
    assert list(manager.uploads_dir.iterdir()) == []


def test_save_uploaded_cleans_up_unparseable_file(manager):
    """解析失败（缺 content）=> 拒绝并删除文件，不让坏文件长期污染热加载。"""
    with pytest.raises(ValueError, match="解析失败"):
        manager.save_uploaded("bad.json", '{"name": "空内容"}')
    assert list(manager.uploads_dir.iterdir()) == []


def test_save_uploaded_rejects_unsupported_suffix(manager):
    """只接受 .md/.txt/.json；被拒文件不落盘（上传目录可能还没创建过）。"""
    with pytest.raises(ValueError, match="仅支持"):
        manager.save_uploaded("evil.exe", "not a skill")
    assert not manager.uploads_dir.exists() or list(manager.uploads_dir.iterdir()) == []


def test_save_uploaded_requires_uploads_dir(tmp_path):
    """未配置上传目录 => RuntimeError（旧调用方不受影响）。"""
    plain = SkillManager(root_dir=str(tmp_path / "skills"))
    with pytest.raises(RuntimeError, match="未配置上传目录"):
        plain.save_uploaded("a.md", "# hi")


def test_summary_include_content_defaults_off(manager):
    """summary 默认不带完整内容，include_content=True 才带（/health 不背大文本）。"""
    summary = manager.summary()
    assert "content" not in summary["skills"][0]
    full = manager.summary(include_content=True)
    # 内置 skill 正文（front matter 已剥掉）
    assert "核验支付凭证" in full["skills"][0]["content"]
    assert full["uploads_dir"] == str(manager.uploads_dir)


# ── /skills/upload 端点（monkeypatch 全局 _skill_manager，不起 FastAPI） ──────

def _make_upload_file(name: str, text: str):
    import io

    from fastapi import UploadFile
    return UploadFile(filename=name, file=io.BytesIO(text.encode("utf-8")))


def test_upload_skill_endpoint_ok(monkeypatch, manager):
    """端点上传成功：返回 skill 摘要，编排器同步热加载。"""
    import api.main as main

    monkeypatch.setattr(main, "_skill_manager", manager)
    calls = {}
    class FakeOrchestrator:
        def set_skill_manager(self, sm):
            calls["manager"] = sm
    monkeypatch.setattr(main, "_orchestrator", FakeOrchestrator())

    # 直接调函数时 agents 必须显式传 None（函数默认值是 FastAPI 的 Form 对象）
    resp = asyncio.run(main.upload_skill(_make_upload_file("新话术.md", "---\nname: 新话术\n---\n内容。\n"), agents=None))
    assert resp["ok"] is True
    assert resp["skill"]["name"] == "新话术"
    assert resp["skill"]["source"] == "uploaded"
    assert calls["manager"] is manager


def test_upload_skill_endpoint_rejects_builtin_name(monkeypatch, manager):
    """端点层面：与内置同名 => 400。"""
    import api.main as main
    from fastapi import HTTPException

    monkeypatch.setattr(main, "_skill_manager", manager)
    monkeypatch.setattr(main, "_orchestrator", None)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.upload_skill(_make_upload_file("x.md", "---\nname: 退款处理\n---\n坏。\n"), agents=None))
    assert exc.value.status_code == 400
    assert "内置" in str(exc.value.detail)


def test_upload_skill_endpoint_503_when_not_ready(monkeypatch):
    """_skill_manager 未初始化 => 503。"""
    import api.main as main
    from fastapi import HTTPException

    monkeypatch.setattr(main, "_skill_manager", None)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.upload_skill(_make_upload_file("a.md", "# hi")))
    assert exc.value.status_code == 503


# ── 上传时选择适用 Agent（agents 参数写进文件） ──────────────────────────────

def test_save_uploaded_injects_agents_into_md_without_frontmatter(manager):
    """md 无 front matter + 指定 agents => 前置生成 front matter。"""
    skill = manager.save_uploaded("新话术.md", "先道歉，再记录诉求。\n", agents=["medication"])
    assert skill.agents == ["medication"]
    saved = (manager.uploads_dir / "新话术.md").read_text(encoding="utf-8")
    assert saved.startswith("---\nagents: medication\n---\n")


def test_save_uploaded_overrides_agents_in_existing_frontmatter(manager):
    """md 自带 agents 行 => 被上传页选择覆盖。"""
    skill = manager.save_uploaded(
        "已有fm.md",
        "---\nname: 已有话术\nagents: appointment\nkeywords: 你好\n---\n正文。\n",
        agents=["symptom_triage", "medication"],
    )
    assert skill.agents == ["symptom_triage", "medication"]
    assert skill.keywords == ["你好"]  # 其他 front matter 行原样保留
    saved = (manager.uploads_dir / "已有fm.md").read_text(encoding="utf-8")
    assert "agents: symptom_triage, medication" in saved
    assert "appointment" not in saved


def test_save_uploaded_empty_agents_means_public(manager):
    """agents=[]（公用）：清掉文件里自带的 agents 限制。"""
    skill = manager.save_uploaded(
        "公用.md",
        "---\nname: 公用话术\nagents: medication\n---\n正文。\n",
        agents=[],
    )
    assert skill.agents == []
    saved = (manager.uploads_dir / "公用.md").read_text(encoding="utf-8")
    assert "agents:" not in saved


def test_save_uploaded_injects_agents_into_json(manager):
    """json 文件：agents 写进 JSON 字段。"""
    skill = manager.save_uploaded(
        "话术.json",
        '{"name": "json话术", "content": "内容。", "agents": ["appointment"]}',
        agents=["emergency"],
    )
    assert skill.agents == ["emergency"]


def test_save_uploaded_agents_none_leaves_file_untouched(manager):
    """agents=None（旧调用方/不传表单）：文件内容不干预，自带 agents 保留。"""
    skill = manager.save_uploaded("旧接口.md", "---\nname: 旧话术\nagents: appointment\n---\n正文。\n")
    assert skill.agents == ["appointment"]


def test_upload_skill_endpoint_passes_agents_form_field(monkeypatch, manager):
    """端点把 agents 表单字段解析成列表传给 save_uploaded。"""
    import api.main as main

    captured = {}
    class RecordingManager:
        def save_uploaded(self, filename, text, agents=None):
            captured["agents"] = agents
            return manager.save_uploaded(filename, text, agents=agents)

    monkeypatch.setattr(main, "_skill_manager", RecordingManager())
    monkeypatch.setattr(main, "_orchestrator", None)
    resp = asyncio.run(main.upload_skill(
        _make_upload_file("范围.md", "---\nname: 范围\n---\n内容。\n"),
        agents="Medication, appointment",
    ))
    assert captured["agents"] == ["medication", "appointment"]
    assert resp["skill"]["agents"] == ["medication", "appointment"]
