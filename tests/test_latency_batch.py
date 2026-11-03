# -*- coding: utf-8 -*-
"""延迟回放脚本的离线单测。"""

from __future__ import annotations

import json
from pathlib import Path

from eval.latency_batch import load_questions


def test_load_questions_from_items(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            {
                "items": [
                    {"qid": "A1", "question": "问题一"},
                    {"qid": "A2", "question": "  "},
                    {"qid": "A3"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert load_questions(path) == [{"qid": "A1", "question": "问题一"}]


def test_load_questions_from_bare_list(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text('[{"qid": "A1", "question": "问题一"}]', encoding="utf-8")
    assert load_questions(path) == [{"qid": "A1", "question": "问题一"}]
