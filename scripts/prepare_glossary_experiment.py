"""第2回発展: 配布スクリプトのQ&Aと独立した評価問題を準備する。API不要。"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "lectures/02/advanced"
SYSTEM = (
    "あなたはAIエンジニアリング講座の用語説明係です。講座の用語集に基づき、日本語で簡潔に説明してください。"
    "用語集に記載のない語については『この講座の用語集には記載がありません。確認が必要です。』と答えてください。"
    "説明文だけを出力してください。"
)
ABSTENTION = "この講座の用語集には記載がありません。確認が必要です。"

# 学習前に固定する簡易採点ルーブリック。各リストから1語以上、全グループを満たす。
RUBRICS = {
    "lora": [["固定", "凍結"], ["差分", "低ランク", "小さ", "行列"]],
    "qlora": [["4bit", "4-bit", "4ビット", "4 bit"], ["lora", "低ランク", "追加"]],
    "quantization": [["ビット", "bit", "精度"], ["メモリ", "容量", "省", "小さ"]],
    "learning-rate": [["更新", "調整", "ステップ", "変化"], ["大き", "幅", "速", "係数", "割合", "どれだけ"]],
    "overfitting": [["学習", "訓練"], ["未知", "汎化", "新しい", "評価", "テスト", "検証", "寄り"]],
    "pretraining": [["大量", "大規模", "汎用"], ["次", "トークン", "予測"]],
    "agent-loop": [["繰り返", "ループ", "反復"], ["ツール", "道具"]],
    "mcp": [["共通", "標準", "規約", "プロトコル"], ["ツール", "外部", "接続"]],
    "lec-start-tag": [["開始", "時点", "スタート", "演習"], ["タグ", "tag"]],
    "homework-levels": [["基礎"], ["実装"], ["発展"], ["1", "一", "いずれ", "どれか"]],
}
PARAPHRASES = {
    "lora": "Low-Rank Adaptationの仕組みを、何を固定し何を学習するかに触れて教えてください。",
    "qlora": "Quantized LoRAは、重みの保持方法と学習する部分をどう組み合わせますか。",
    "quantization": "quantizationをすると、数値の表現と必要な記憶容量はどう変わりますか。",
    "learning-rate": "learning rateはパラメータ更新の何を調節する設定ですか。",
    "overfitting": "オーバーフィッティングとは、手元のデータと新しいデータで成績がどう違う状態ですか。",
    "pretraining": "プレトレーニングでは、どんなデータで何を予測するように学びますか。",
    "agent-loop": "ツール呼び出しループでは、AIはどんな処理を繰り返しますか。",
    "mcp": "Model Context Protocolは、何と何を接続するための取り決めですか。",
    "lec-start-tag": "開始タグは、この講座で何のために使いますか。",
    "homework-levels": "この講座の3レベルの宿題は、全部やる必要がありますか。区分も教えてください。",
}
DESCRIPTIONS = {
    "lora": "大きなモデルの元の係数を変えず、細長い二つの行列だけで追加学習する方法を説明してください。",
    "quantization": "モデルの数値を少ない桁数で保持し、記憶容量を節約する技術を説明してください。",
    "overfitting": "練習で見たデータに適応しすぎて、初めての入力で成績が落ちる現象を説明してください。",
    "agent-loop": "AIが次の行動を決め、道具の結果を受け取り、また次の行動を決める流れを説明してください。",
    "lec-start-tag": "前の回の作業が終わっていなくても、今回の演習の初期状態から始めるための仕組みを説明してください。",
}
TRAIN_UNKNOWN = [
    "バッチ正規化", "カーネル法", "ベイズ最適化", "主成分分析",
    "サポートベクターマシン", "変分自己符号化器", "星雲折り畳み学習", "透明ねこ勾配法",
]
TEST_UNKNOWN = [
    "拡散モデル", "勾配クリッピング", "決定木", "自己組織化マップ", "確率的主成分分析", "カプセルネットワーク",
    "月光整流学習法", "銀河しおり最適化", "雲上ねじれ推論", "青空コンパス学習",
]


def dump_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def prepare() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = subprocess.check_output([sys.executable, str(ROOT / "scripts/make_sft_dataset.py")], text=True)
    (OUT / "glossary_sft.jsonl").write_text(raw, encoding="utf-8")
    originals = [json.loads(line) for line in raw.splitlines() if line.strip()]
    terms = json.loads((ROOT / "data/glossary.json").read_text(encoding="utf-8"))["terms"]
    by_id = {t["id"]: t for t in terms}
    train = [{"messages": [{"role": "system", "content": SYSTEM}] + r["messages"]} for r in originals]
    for term in TRAIN_UNKNOWN:
        for q in [f"「{term}」とは何ですか？", f"{term}について簡単に説明してください。"]:
            train.append({"messages": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": q},
                                      {"role": "assistant", "content": ABSTENTION}]})
    valid = [{"messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": f"用語『{t['term']}』の定義を教えてください。"},
                           {"role": "assistant", "content": t["definition"]}]} for t in terms]
    evaluation = []
    for term_id, groups in RUBRICS.items():
        t = by_id[term_id]
        questions = [("direct", f"講座用語『{t['term']}』の意味を説明してください。"),
                     ("paraphrase", PARAPHRASES[term_id])]
        if term_id in DESCRIPTIONS:
            questions.append(("description", DESCRIPTIONS[term_id]))
        for category, question in questions:
            evaluation.append({"id": f"{category}:{term_id}", "category": category,
                               "question": question, "term_id": term_id, "term": t["term"],
                               "reference": t["definition"], "expected_abstention": False,
                               "concept_groups": groups})
    for i, term in enumerate(TEST_UNKNOWN):
        evaluation.append({"id": f"unknown:{i:02d}", "category": "unknown_real" if i < 6 else "unknown_fictional",
                           "question": f"講座用語『{term}』の意味を説明してください。", "term_id": None, "term": term,
                           "reference": ABSTENTION, "expected_abstention": True, "concept_groups": []})
    train_q = {r["messages"][-2]["content"] for r in train}
    valid_q = {r["messages"][-2]["content"] for r in valid}
    test_q = {r["question"] for r in evaluation}
    assert len(train_q) == len(train), "duplicate train questions"
    assert len(valid_q) == len(valid), "duplicate valid questions"
    assert len(test_q) == len(evaluation), "duplicate evaluation questions"
    assert not (train_q & valid_q or train_q & test_q or valid_q & test_q), "question leakage"
    dump_jsonl(OUT / "train_messages.jsonl", train)
    dump_jsonl(OUT / "valid_messages.jsonl", valid)
    # 正解を含む評価データは生成関数に渡さず、採点時だけ参照する。
    dump_jsonl(OUT / "evaluation_gold.jsonl", evaluation)
    files = ["glossary_sft.jsonl", "train_messages.jsonl", "valid_messages.jsonl", "evaluation_gold.jsonl"]
    manifest = {"glossary_terms": len(terms), "raw_positive_examples": len(originals),
                "train_positive": len(originals), "train_abstention": len(train)-len(originals),
                "train_total": len(train), "valid": len(valid), "evaluation": len(evaluation),
                "evaluation_categories": {c: sum(r["category"] == c for r in evaluation) for c in sorted({r["category"] for r in evaluation})},
                "split": "同じ用語の定義は共有するが質問文はtrain/valid/testで非重複。評価問題とルーブリックは学習前に固定。",
                "sha256": {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in files}}
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    prepare()
