"""rebuild_full_index 分块参数透传（L5-A）：--min-chars 语义单测（零外部依赖）。

正式数据语料（CORPUS_ROOT）不入库，缺失时整体跳过；本地存在则真实校验透传。
"""

import pytest

from tools.data_scripts.rebuild_full_index import CORPUS_ROOT, build_config_with_split

pytestmark = pytest.mark.skipif(not CORPUS_ROOT.is_dir(), reason="正式数据语料仅本地存在（不入库）")


def test_min_chars_default_takes_config() -> None:
    """min_chars=-1 时取配置默认 200（开启碎片合并）。"""
    assert build_config_with_split("t", min_chars=-1).CHUNK_MIN_CHARS == 200


def test_min_chars_zero_disables_merge() -> None:
    """min_chars=0 时显式关闭合并（对照旧行为）。"""
    assert build_config_with_split("t", min_chars=0).CHUNK_MIN_CHARS == 0


def test_min_chars_other_value_overrides() -> None:
    """min_chars>=0 时按传入值覆盖。"""
    assert build_config_with_split("t", min_chars=300).CHUNK_MIN_CHARS == 300


def test_chunk_size_and_overlap_still_passthrough() -> None:
    """原有 --chunk-size / --chunk-overlap 语义不变。"""
    cfg = build_config_with_split("t", chunk_size=512, chunk_overlap=150)
    assert cfg.CHUNK_SIZE == 512
    assert cfg.CHUNK_OVERLAP == 150
    assert build_config_with_split("t", chunk_size=0).CHUNK_SIZE == 1024
