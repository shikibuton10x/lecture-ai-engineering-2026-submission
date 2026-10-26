"""第1回・発展: 履歴なしのGemini呼び出しを計測し、生データと集計を保存する。

例: uv run --env-file .env python -m app.benchmark --trials 3
APIを使わず計画だけ確認: uv run python -m app.benchmark --dry-run
"""

import argparse
import csv
import json
import platform
import random
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import httpx
from google import genai
from google.genai import errors, types

from app.config import get_api_key

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "lectures/01/benchmark_questions.json"
DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
TOKEN_FIELDS = {
    "input_tokens": "prompt_token_count",
    "output_tokens": "candidates_token_count",
    "thinking_tokens": "thoughts_token_count",
    "total_tokens": "total_token_count",
    "cached_tokens": "cached_content_token_count",
}


def make_plan(models, questions, trials, seed):
    """実験順を固定seedで混ぜ、モデルと実行時刻の偏りを減らす。"""
    plan = [
        {"model": model, "question_id": q["id"], "prompt": q["prompt"], "trial": trial}
        for trial in range(1, trials + 1)
        for q in questions
        for model in models
    ]
    random.Random(seed).shuffle(plan)
    return plan


def measure(generate, item, max_output_tokens, clock=time.perf_counter):
    """SDKの呼び出し開始〜返却を測る。待機やファイル保存の時間は含まない。"""
    row = {
        **item,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "error_code": None,
        "error_type": None,
        "model_version": None,
        "finish_reason": None,
        "response_text": None,
        **dict.fromkeys(TOKEN_FIELDS),
    }
    started = clock()
    try:
        response = generate(
            model=item["model"],
            contents=item["prompt"],
            config=types.GenerateContentConfig(max_output_tokens=max_output_tokens),
        )
    except (errors.APIError, httpx.HTTPError) as exc:
        row["latency_sec"] = clock() - started
        row["status"] = "api_error" if isinstance(exc, errors.APIError) else "transport_error"
        row["error_code"] = getattr(exc, "code", None)
        # 生の例外文・HTTPヘッダーには認証情報が含まれ得るため保存しない。
        row["error_type"] = type(exc).__name__
        return row
    row["latency_sec"] = clock() - started
    row["model_version"] = response.model_version
    row["response_text"] = response.text
    if response.candidates:
        reason = response.candidates[0].finish_reason
        row["finish_reason"] = getattr(reason, "value", reason)
    usage = response.usage_metadata
    for field, attribute in TOKEN_FIELDS.items():
        row[field] = getattr(usage, attribute, None)
    if row["finish_reason"] == "MAX_TOKENS":
        row["status"] = "truncated"
    elif not row["response_text"]:
        row["status"] = "empty"
    elif row["finish_reason"] != "STOP":
        row["status"] = "incomplete"
    return row


def summarize(rows):
    """完了回答だけを集計する。欠損した使用量を0で補わない。"""
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"], row["question_id"]].append(row)
    summaries = []
    for (model, question_id), group in sorted(groups.items()):
        ok = [r for r in group if r["status"] == "ok"]
        result = {
            "model": model,
            "question_id": question_id,
            "attempts": len(group),
            "successes": len(ok),
            "non_successes": len(group) - len(ok),
        }
        for field in ["latency_sec", *TOKEN_FIELDS]:
            values = [r[field] for r in ok if r[field] is not None]
            result[field + "_n"] = len(values)
            result[field + "_mean"] = statistics.mean(values) if values else None
            if field == "latency_sec":
                result[field + "_median"] = statistics.median(values) if values else None
                result[field + "_min"] = min(values) if values else None
                result[field + "_max"] = max(values) if values else None
        summaries.append(result)
    return summaries


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_summaries(directory, rows):
    summary = summarize(rows)
    write_csv(directory / "measurements.csv", rows)
    write_csv(directory / "summary.csv", summary)

    def number(value):
        return "—" if value is None else f"{value:.3f}"

    lines = [
        "# 第1回 発展課題・実測集計",
        "",
        "`raw.jsonl` の完了回答（status=ok）から生成。失敗・空回答・打ち切りは平均から除外。",
        "",
        "| モデル | 質問 | 成功/試行 | 平均秒 | 中央値秒 | 平均入力token | 平均出力token | 平均推論token |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['question_id']} | {row['successes']}/{row['attempts']} | "
            f"{number(row['latency_sec_mean'])} | {number(row['latency_sec_median'])} | "
            f"{number(row['input_tokens_mean'])} | {number(row['output_tokens_mean'])} | "
            f"{number(row['thinking_tokens_mean'])} |"
        )
    lines += [
        "",
        "欠損値は — と表示。各指標の標本数は summary.csv の *_n を参照。",
        "応答時間はSDKの返却までの時間であり、ネットワーク・推論・生成などを含む。",
        "推論設定はモデル既定値。少数回の観察で性能の優劣や有意差は断定できない。",
        "",
    ]
    (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(plan, generate, directory, max_output_tokens, interval, sleep=time.sleep):
    rows = []
    try:
        # 途中終了しても完了した試行のログを失わないよう1行ずつ保存する。
        with (directory / "raw.jsonl").open("x", encoding="utf-8") as handle:
            for index, item in enumerate(plan, 1):
                if index > 1:
                    sleep(interval)
                row = measure(generate, {**item, "request_index": index}, max_output_tokens)
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{index}/{len(plan)}] {row['model']} {row['question_id']} "
                    f"trial={row['trial']} {row['status']} {row['latency_sec']:.3f}s",
                    flush=True,
                )
                # キー・モデル・クオータの問題を繰り返し送信しない。
                if row["error_code"] in (400, 401, 403, 404, 429):
                    print(f"HTTP {row['error_code']} のため中断。設定・利用上限を確認してください。")
                    break
    finally:
        save_summaries(directory, rows)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--interval", type=float, default=20, help="呼び出し終了後の待機秒数")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="計画を表示。API呼び出しなし")
    parser.add_argument("--list-models", action="store_true", help="生成可能なモデル名を表示して終了")
    args = parser.parse_args(argv)
    if args.trials < 1 or args.max_output_tokens < 1:
        parser.error("trials と max-output-tokens は1以上にしてください")
    if args.interval < 0 or args.timeout_seconds <= 0:
        parser.error("interval は0以上、timeout-seconds は正数にしてください")
    if len(set(args.models)) != len(args.models):
        parser.error("モデル名が重複しています")
    return args


def main(argv=None):
    args = parse_args(argv)
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise SystemExit("質問は1件以上、idは一意にしてください。")
    if any(not q.get("prompt", "").strip() for q in questions):
        raise SystemExit("空の質問は使用できません。")
    plan = make_plan(args.models, questions, args.trials, args.seed)
    settings = {
        "models": args.models,
        "trials": args.trials,
        "planned_requests": len(plan),
        "interval_sec": args.interval,
        "max_output_tokens": args.max_output_tokens,
        "timeout_sec": args.timeout_seconds,
        "seed": args.seed,
        "sdk_attempts": 1,
        "temperature": "model_default",
        "thinking": "model_default",
        "conversation_history": False,
        "warmup_requests": 0,
        "questions": questions,
    }
    if args.dry_run:
        print(json.dumps({"settings": settings, "plan": plan}, ensure_ascii=False, indent=2))
        return 0

    # キーは環境変数だけから受け取る。設定・結果ファイルには保存しない。
    with genai.Client(
        api_key=get_api_key(),
        http_options=types.HttpOptions(
            timeout=int(args.timeout_seconds * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    ) as client:
        if args.list_models:
            for model in client.models.list():
                if "generateContent" in (model.supported_actions or []):
                    print(model.name)
            return 0

        directory = args.out or ROOT / "lectures/01/results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory.mkdir(parents=True, exist_ok=False)
        metadata = {
            "settings": settings,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "google_genai_version": version("google-genai"),
            "run_status": "running",
        }
        metadata_path = directory / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"結果保存先: {directory}", flush=True)
        try:
            rows = run(plan, client.models.generate_content, directory, args.max_output_tokens, args.interval)
        except KeyboardInterrupt:
            metadata["run_status"] = "interrupted"
            return 130
        except Exception:
            metadata["run_status"] = "failed"
            raise
        else:
            complete = len(rows) == len(plan) and all(r["status"] == "ok" for r in rows)
            metadata["run_status"] = "completed" if complete else "incomplete"
            return 0 if complete else 1
        finally:
            metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
