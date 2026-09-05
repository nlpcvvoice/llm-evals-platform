"""Tests for GoldenSet loading, validation, and sampling."""

from __future__ import annotations

import json
import random

import pytest

from llm_evals.eval import DatasetError, GoldenSet, QAItem, as_dict, from_dict
from llm_evals.eval.validation import validate_item

DATA_DIR = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "golden"


def _item(item_id: str, **overrides) -> dict[str, object]:
    base = {
        "id": item_id,
        "question": "question?",
        "reference_answer": "answer",
        "context": [],
        "expected_evals": [],
        "metadata": {},
    }
    base.update(overrides)
    return base


def _write(path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- loading from JSONL ---


def test_from_jsonl_valid(tmp_path):
    p = tmp_path / "d.jsonl"
    _write(p, [json.dumps(_item("a-1")), json.dumps(_item("a-2"))])
    gs = GoldenSet.from_jsonl(p)
    assert gs.size == 2
    assert [it.id for it in gs.items] == ["a-1", "a-2"]


def test_from_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "d.jsonl"
    _write(p, [json.dumps(_item("a-1")), "", json.dumps(_item("a-2"))])
    gs = GoldenSet.from_jsonl(p)
    assert gs.size == 2


def test_from_jsonl_missing_file(tmp_path):
    with pytest.raises(DatasetError):
        GoldenSet.from_jsonl(tmp_path / "nope.jsonl")


def test_from_jsonl_bad_json(tmp_path):
    p = tmp_path / "d.jsonl"
    _write(p, ["{not json", json.dumps(_item("a-1"))])
    with pytest.raises(DatasetError, match="invalid item"):
        GoldenSet.from_jsonl(p)


def test_from_jsonl_rejects_invalid_item(tmp_path):
    p = tmp_path / "d.jsonl"
    _write(p, [json.dumps(_item("a-1", question=""))])  # empty question
    with pytest.raises(DatasetError, match="invalid"):
        GoldenSet.from_jsonl(p)


def test_duplicate_id_rejected():
    gs = GoldenSet()
    gs.add(QAItem("x", "q", "a"))
    with pytest.raises(DatasetError, match="duplicate"):
        gs.add(QAItem("x", "q2", "a2"))


def test_empty_jsonl_rejected(tmp_path):
    p = tmp_path / "d.jsonl"
    _write(p, [])
    with pytest.raises(DatasetError, match="empty"):
        GoldenSet.from_jsonl(p)


# --- validation ---


def test_valid_item_passes():
    result = validate_item(QAItem("ok-1", "q", "a"))
    assert result.ok
    assert not result.errors


def test_empty_question_fails():
    result = validate_item(QAItem("ok-1", "  ", "a"))
    assert not result.ok
    assert any("question" in e for e in result.errors)


def test_bad_id_fails():
    result = validate_item(QAItem("BAD ID!", "q", "a"))
    assert not result.ok
    assert any("id" in e for e in result.errors)


def test_unknown_eval_kind_fails():
    result = validate_item(QAItem("ok-1", "q", "a", expected_evals=["nonsense"]))
    assert not result.ok
    assert any("unknown eval kind" in e for e in result.errors)


# --- sampling & stats ---


def test_sample_size():
    gs = GoldenSet()
    for i in range(10):
        gs.add(QAItem(f"x-{i}", f"q{i}", "a"))
    assert len(gs.sample(3)) == 3
    assert len(gs.sample(0)) == 0
    assert len(gs.sample(100)) == 10


def test_sample_no_replacement():
    gs = GoldenSet()
    for i in range(5):
        gs.add(QAItem(f"x-{i}", f"q{i}", "a"))
    ids = [it.id for it in gs.sample(5, rng=random.Random(42))]
    assert sorted(ids) == [f"x-{i}" for i in range(5)]


def test_stats():
    gs = GoldenSet()
    gs.add(QAItem("a", "q", "ans", context=["c"], expected_evals=["faithfulness"]))
    gs.add(QAItem("b", "q", "ans", expected_evals=["recall"]))
    stats = gs.stats()
    assert stats["total"] == 2
    assert stats["with_context"] == 1


# --- round-trip serialization ---


def test_as_dict_from_dict_roundtrip():
    item = QAItem(
        "rt-1",
        "question?",
        "answer",
        context=["ctx"],
        expected_evals=["faithfulness"],
        metadata={"k": "v"},
    )
    raw = as_dict(item)
    restored = from_dict(raw)
    assert restored == item
    assert as_dict(restored) == raw


def test_golden_on_disk_loads():
    gs = GoldenSet.from_jsonl(DATA_DIR / "seed.jsonl")
    assert gs.size == 5
    stats = gs.stats()
    assert stats["total"] == 5
