"""
查询别名展开的单元测试（P2）。

覆盖：
1. 命中展开：查询含 from 词 → 追加语料规范名（保留原词）；
2. 多组命中全部追加 + 去重；
3. 有向性：只有 from 触发 to，反过来不触发（上位词不追加下位叫法）；
4. 无匹配原样返回、文件缺失降级为空表（检索链路不因别名表故障而中断）。

全部离线：只读 data/knowledge/aliases.json 与纯函数计算，不连 ChromaDB、不加载模型。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mcp.knowledge_base as kb


def _reset_cache(monkeypatch) -> None:
    """每个用例前清掉模块级惰性缓存，保证重新从文件加载。"""
    monkeypatch.setattr(kb, "_aliases_cache", None)


def test_expand_alias_appends_canonical_word(monkeypatch):
    """贯叶连翘 → 追加圣约翰草；原词保留（原词对向量路仍有价值）。"""
    _reset_cache(monkeypatch)
    out = kb._expand_aliases("贯叶连翘和华法林能一起吃吗")
    assert out.startswith("贯叶连翘和华法林能一起吃吗")
    assert "圣约翰草" in out


def test_expand_multiple_groups_all_append(monkeypatch):
    """一段查询同时命中多组 from，各组 to 全部追加。"""
    _reset_cache(monkeypatch)
    out = kb._expand_aliases("吃高丽参和白参要紧吗")
    assert "人参" in out


def test_no_match_returns_unchanged(monkeypatch):
    """没有别名词的查询原样返回（不无谓加词干扰检索）。"""
    _reset_cache(monkeypatch)
    assert kb._expand_aliases("华法林和布洛芬能一起吃吗") == "华法林和布洛芬能一起吃吗"


def test_expansion_is_directional(monkeypatch):
    """有向性：搜「甘草」不追加「甘草胶囊」；搜「甘草胶囊」才追加「甘草」。"""
    _reset_cache(monkeypatch)
    assert kb._expand_aliases("甘草和螺内酯") == "甘草和螺内酯"
    assert kb._expand_aliases("甘草胶囊和螺内酯").endswith("甘草")


def test_missing_alias_file_degrades_to_empty(monkeypatch):
    """别名文件缺失时返回空表，展开变成原样返回（检索链路不受影响）。"""
    monkeypatch.setattr(kb, "_ALIASES_PATH", Path("data/knowledge/aliases_不存在.json"))
    _reset_cache(monkeypatch)
    assert kb._load_aliases() == []
    assert kb._expand_aliases("贯叶连翘") == "贯叶连翘"
