"""実APIを使わず、計測・集計の誤りとログ保存を検証する。"""

import json
from types import SimpleNamespace

import pytest
from google.genai import errors, types

from app.benchmark import make_plan, measure, run, summarize


def response(text="回答", reason="STOP", usage=True):
    return SimpleNamespace(
        text=text,
        model_version="test-model-version",
        candidates=[SimpleNamespace(finish_reason=reason)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=20,
            thoughts_token_count=30,
            total_token_count=60,
        ) if usage else None,
    )


ITEM = {"model": "test-model", "question_id": "q1", "prompt": "質問", "trial": 1}


def measured(result):
    clock = iter([100.0, 102.5])
    return measure(lambda **kwargs: result, ITEM, 2048, clock=lambda: next(clock))


def test_plan_covers_each_condition_and_is_reproducible():
    questions = [{"id": "a", "prompt": "A"}, {"id": "b", "prompt": "B"}]
    plan = make_plan(["m1", "m2"], questions, 3, 42)
    assert plan == make_plan(["m1", "m2"], questions, 3, 42)
    assert len(plan) == 12
    assert len({(r["model"], r["question_id"], r["trial"]) for r in plan}) == 12


def test_measure_keeps_thinking_and_total_tokens_separate():
    row = measured(response())
    assert row["latency_sec"] == 2.5
    assert row["input_tokens"] == 10
    assert row["output_tokens"] == 20
    assert row["thinking_tokens"] == 30
    assert row["total_tokens"] == 60
    assert row["cached_tokens"] is None


def test_missing_usage_is_not_reported_as_zero():
    row = measured(response(usage=False))
    summary = summarize([row])[0]
    assert row["input_tokens"] is None
    assert summary["input_tokens_n"] == 0
    assert summary["input_tokens_mean"] is None
    assert summary["latency_sec_mean"] == 2.5


@pytest.mark.parametrize("result,status", [
    (response(reason="MAX_TOKENS"), "truncated"),
    (response(text=None), "empty"),
    (response(reason="SAFETY"), "incomplete"),
])
def test_non_completed_responses_are_excluded_from_success_means(result, status):
    good, bad = measured(response()), measured(result)
    bad["latency_sec"] = 100
    assert bad["status"] == status
    summary = summarize([good, bad])[0]
    assert summary["attempts"] == 2
    assert summary["successes"] == 1
    assert summary["latency_sec_mean"] == 2.5


def test_quota_error_stops_requests_and_preserves_error_log(tmp_path):
    calls = []

    def quota_error(**kwargs):
        calls.append(kwargs)
        raise errors.ClientError(429, {"error": {"message": "secret-should-not-be-saved"}})

    rows = run([ITEM, ITEM], quota_error, tmp_path, 2048, 0)
    assert len(calls) == 1
    assert rows[0]["error_code"] == 429
    raw = (tmp_path / "raw.jsonl").read_text()
    assert "secret-should-not-be-saved" not in raw
    assert json.loads(raw)["status"] == "api_error"
    assert (tmp_path / "summary.csv").is_file()
    assert summarize(rows)[0]["latency_sec_mean"] is None


def test_interrupt_keeps_completed_samples_and_saves_summary(tmp_path):
    calls = iter([response(), KeyboardInterrupt()])

    def generate(**kwargs):
        result = next(calls)
        if isinstance(result, BaseException):
            raise result
        return result

    with pytest.raises(KeyboardInterrupt):
        run([ITEM, ITEM], generate, tmp_path, 2048, 0)
    raw = (tmp_path / "raw.jsonl").read_text().splitlines()
    assert len(raw) == 1
    assert json.loads(raw[0])["status"] == "ok"
    assert (tmp_path / "summary.md").is_file()
