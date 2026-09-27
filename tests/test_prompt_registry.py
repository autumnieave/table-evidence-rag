# -*- coding: utf-8 -*-
"""B-13 prompts 版本注册表一致性单测：零外部依赖（不调 LLM/MySQL）。

校验 prompts/registry.json 与各业务模块版本常量保持一致，防止「改了模板忘 bump
版本号 / registry 漏登记」的漂移；与 tests/test_metric_standardize.py 的版本断言互补。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import prompts.agent
import prompts.fallback
import prompts.financial
import prompts.multi_agent
import prompts.pipeline
import prompts.rag
from prompts import PROMPT_VERSION

_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY_PATH = _ROOT / "src" / "prompts" / "registry.json"


def _load_registry() -> dict:
    with _REGISTRY_PATH.open(encoding="utf-8") as f:
        return json.load(f)


# module 键 -> 代码内真实版本常量（与 registry.json 逐条对照）
_EXPECTED = {
    "__init__.py": PROMPT_VERSION,
    "agent": prompts.agent.AGENT_PROMPT_VERSION,
    "rag": prompts.rag.RAG_PROMPT_VERSION,
    "pipeline": prompts.pipeline.PIPELINE_PROMPT_VERSION,
    "multi_agent": prompts.multi_agent.MULTI_AGENT_PROMPT_VERSION,
    "financial": prompts.financial.FINANCIAL_PROMPT_VERSION,
    "fallback": prompts.fallback.FALLBACK_PROMPT_VERSION,
}


def test_registry_entries_cover_all_prompt_modules() -> None:
    """registry 登记的模块集合必须与代码常量集合一致（不多不少）。"""
    reg = _load_registry()
    modules = {entry["module"] for entry in reg["templates"]}
    assert modules == set(_EXPECTED), (
        "registry 模块集合与代码不一致，缺失或多余: "
        f"{modules ^ set(_EXPECTED)}"
    )


def test_registry_version_matches_module_constants() -> None:
    """每条 registry 记录的 version 必须等于对应模块的版本常量。"""
    reg = _load_registry()
    by_module = {entry["module"]: entry for entry in reg["templates"]}
    for module, version in _EXPECTED.items():
        assert by_module[module]["version"] == version, (
            f"{module} 不一致：registry={by_module[module]['version']}，代码常量={version}"
        )


def test_registry_required_fields_and_format() -> None:
    """登记字段齐全、版本号符合 yyyy-mm-dd-vN 格式、template_id/file 非空。"""
    reg = _load_registry()
    required = {"template_id", "module", "file", "version", "updated_at", "changelog", "eval_ref"}
    version_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}-v\d+$")
    for entry in reg["templates"]:
        assert required <= set(entry), f"缺字段: {required - set(entry)} @ {entry}"
        assert version_pattern.match(entry["version"]), f"版本格式非法: {entry}"
        assert entry["template_id"].startswith("prompt."), entry
        assert entry["file"].endswith(".py"), entry


def test_package_version_is_latest_registered_touch() -> None:
    """包级版本应为 registry 中 __init__.py 条目的 version。"""
    reg = _load_registry()
    package_entry = next(e for e in reg["templates"] if e["module"] == "__init__.py")
    assert package_entry["version"] == PROMPT_VERSION
