"""Colab用の段階実行可能な用語集SFT比較。生成とgoldの参照を分離する。"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "lectures/02/advanced"
SYSTEM = (
    "あなたはAIエンジニアリング講座の用語説明係です。講座の用語集に基づき、日本語で簡潔に説明してください。"
    "用語集に記載のない語については『この講座の用語集には記載がありません。確認が必要です。』と答えてください。"
    "説明文だけを出力してください。"
)
ABSTENTION = "この講座の用語集には記載がありません。確認が必要です。"
CONFIG = {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42, "num_train_epochs": 10, "lr": 2e-4,
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "max_seq_len": 512, "max_new_tokens": 192,
    "batch_size": 4, "gradient_accumulation_steps": 2,
    "warmup_steps": 17, "do_sample": False,
    "method": "LoRA, FP32, no quantization",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", text.casefold())


def is_abstention(text: str) -> bool:
    t = normalized(text)
    return "用語集" in t and any(x in t for x in [
        "記載がありません", "記載されていません", "見つかりません", "含まれていません",
        "確認が必要", "載っていません", "掲載されていません", "登録されていません",
    ])


def extract_search_query(question: str, terms: list[dict]) -> str:
    """質問文だけから検索語を得る。評価の正解ラベルは使わない。"""
    quoted = re.search(r"[『「]([^』」]+)[』」]", question)
    if quoted:
        return quoted.group(1)
    q = normalized(question)
    matches = []
    for entry in terms:
        for name in [entry["term"], entry["id"], *entry["aliases"]]:
            # 1文字のr等を自然文の任意箇所に当てない。
            if len(normalized(name)) >= 3 and normalized(name) in q:
                matches.append((len(normalized(name)), name))
    return max(matches)[1] if matches else question


def score_answer(text: str, gold: dict) -> dict:
    abstained = is_abstention(text)
    groups = gold["concept_groups"]
    matched = [any(normalized(word) in normalized(text) for word in group) for group in groups]
    correct = abstained if gold["expected_abstention"] else bool(text.strip()) and not abstained and all(matched)
    return {"abstained": abstained, "concept_group_matches": matched,
            "keyword_content_ok": bool(correct), "expected_abstention": gold["expected_abstention"]}


def summarize(rows: list[dict], method: str) -> dict:
    known = [r for r in rows if not r["expected_abstention"]]
    unknown = [r for r in rows if r["expected_abstention"]]
    categories = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row)
    return {
        "method": method, "n": len(rows),
        "keyword_policy_rate": sum(r["keyword_content_ok"] for r in rows)/len(rows),
        "known_keyword_rate": sum(r["keyword_content_ok"] for r in known)/len(known),
        "known_abstention_rate": sum(r["abstained"] for r in known)/len(known),
        "unknown_abstention_rate": sum(r["abstained"] for r in unknown)/len(unknown),
        "median_seconds": statistics.median(r["seconds"] for r in rows),
        "total_seconds": sum(r["seconds"] for r in rows),
        "mean_output_tokens": statistics.mean(r.get("output_tokens", 0) for r in rows),
        "by_category": {k: {"n": len(v), "keyword_policy_rate": sum(r["keyword_content_ok"] for r in v)/len(v)} for k, v in categories.items()},
    }


class Experiment:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        self.torch = torch
        set_seed(CONFIG["seed"])
        assert torch.cuda.is_available(), "This experiment requires a Colab GPU."
        self.device = "cuda"
        self.gold = read_jsonl(OUT / "evaluation_gold.jsonl")
        self.train_records = read_jsonl(OUT / "train_messages.jsonl")
        self.valid_records = read_jsonl(OUT / "valid_messages.jsonl")
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            CONFIG["base_model"], revision=CONFIG["revision"], dtype=torch.float32).to(self.device)
        self.model.eval()
        self.model.config.use_cache = True
        # 配布モデルのEOSとtokenizerの会話終端は異なる。Trainerによる自動変更の前から統一する。
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.results = {}
        self.training = None
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from app.tools import search_glossary
        self.search = search_glossary
        self.terms = json.loads((ROOT / "data/glossary.json").read_text(encoding="utf-8"))["terms"]
        self.environment = {
            "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
            "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory/1e9, 3),
            "base_weight_gb": round(self.model.get_memory_footprint()/1e9, 3),
            "generation_eos_token_id": self.tokenizer.eos_token_id,
            "generation_pad_token_id": self.tokenizer.pad_token_id,
            "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "trl", "datasets", "accelerate"]},
        }
        # transformers 5ではtokenize=Trueの戻り値が辞書型になるため、明示的にinput_idsを数える。
        lengths = [len(self.tokenizer(self.tokenizer.apply_chat_template(
            r["messages"], tokenize=False, enable_thinking=False))["input_ids"]) for r in self.train_records]
        self.environment["train_sequence_length"] = {"max": max(lengths), "median": statistics.median(lengths),
                                                     "over_limit": sum(n > CONFIG["max_seq_len"] for n in lengths)}
        assert max(lengths) <= CONFIG["max_seq_len"], "Training sequence would be truncated."
        self.verify_sources()
        write_json(OUT / "environment.json", self.environment)
        write_json(OUT / "config.json", CONFIG)
        print(json.dumps(self.environment, ensure_ascii=False, indent=2))

    def verify_sources(self):
        manifest = json.loads((OUT / "dataset_manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["sha256"].items():
            assert hashlib.sha256((OUT/name).read_bytes()).hexdigest() == expected, name
        t = {r["messages"][-2]["content"] for r in self.train_records}
        v = {r["messages"][-2]["content"] for r in self.valid_records}
        e = {r["question"] for r in self.gold}
        assert not (t & v or t & e or v & e)
        print("Dataset hashes and question split verified.")

    def generate(self, question: str, context: str | None = None) -> tuple[str, int]:
        # gold/reference/term_idはこの関数に渡さない。
        user = question if context is None else f"検索された講座用語集の記載:\n{context}\n\n質問: {question}"
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"], do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      eos_token_id=self.tokenizer.eos_token_id)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), len(new_tokens)

    def retrieve(self, question: str) -> tuple[str, list[dict]]:
        raw = self.search(extract_search_query(question, self.terms))
        try:
            hits = json.loads(raw)
        except json.JSONDecodeError:
            hits = []
        return raw, hits

    def evaluate(self, method: str) -> dict:
        assert method in {"base_closedbook", "search_only", "base_rag", "sft_closedbook"}
        if method == "sft_closedbook":
            assert self.training is not None, "Run training first."
        if method in {"base_closedbook", "base_rag"}:
            assert self.training is None, "Run the base comparisons before inserting adapters."
        rows = []
        if method != "search_only":
            self.generate("LoRAを簡単に説明してください。")  # 初回初期化の影響を減らす。採点対象外。
        for i, gold in enumerate(self.gold, 1):
            t0 = time.perf_counter()
            retrieved, hits = "", []
            output_tokens = 0
            if method == "search_only":
                retrieved, hits = self.retrieve(gold["question"])
                answer = "\n".join(h["definition"] for h in hits) if hits else ABSTENTION
            elif method == "base_rag":
                retrieved, hits = self.retrieve(gold["question"])
                context = "\n".join(f"{h['term']}: {h['definition']}" for h in hits) if hits else "該当する項目は見つかりませんでした。"
                answer, output_tokens = self.generate(gold["question"], context=context)
            else:
                answer, output_tokens = self.generate(gold["question"])
            elapsed = time.perf_counter()-t0
            row = {"id": gold["id"], "category": gold["category"], "question": gold["question"],
                   "reference": gold["reference"], "answer": answer, "seconds": round(elapsed, 5),
                   "output_tokens": output_tokens, "retrieved_terms": [h["term"] for h in hits],
                   "search_query": extract_search_query(gold["question"], self.terms) if method in {"search_only", "base_rag"} else None,
                   **score_answer(answer, gold)}
            rows.append(row)
            # 中断時も終わった質問までの原文を保存する。
            write_json(OUT / f"eval_{method}.json", {"method": method, "complete": False, "rows": rows})
            print(f"{method} {i}/{len(self.gold)} {gold['id']}: {int(row['keyword_content_ok'])} ({elapsed:.1f}s)", flush=True)
        summary = summarize(rows, method)
        write_json(OUT / f"eval_{method}.json", {"method": method, "complete": True, "summary": summary, "rows": rows})
        self.results[method] = {"summary": summary, "rows": rows}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    def train(self) -> dict:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
        self.torch.cuda.reset_peak_memory_stats()
        self.model.config.use_cache = False
        lora = LoraConfig(r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=CONFIG["lora_dropout"],
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        args = SFTConfig(output_dir=str(OUT / "checkpoints"), num_train_epochs=CONFIG["num_train_epochs"],
                         per_device_train_batch_size=CONFIG["batch_size"], per_device_eval_batch_size=4,
                         gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"], learning_rate=CONFIG["lr"],
                         lr_scheduler_type="cosine", warmup_steps=CONFIG["warmup_steps"], logging_steps=10,
                         save_strategy="epoch", save_total_limit=2, fp16=False,
                         max_length=CONFIG["max_seq_len"], packing=False, report_to="none", seed=CONFIG["seed"])
        trainer = SFTTrainer(model=self.model, args=args, train_dataset=Dataset.from_list(self.train_records),
                             eval_dataset=Dataset.from_list(self.valid_records), peft_config=lora, processing_class=self.tokenizer)
        trainer.model.print_trainable_parameters()
        trainable, total = trainer.model.get_nb_trainable_parameters()
        t0 = time.perf_counter()
        outcome = trainer.train()
        train_seconds = time.perf_counter()-t0
        validation = trainer.evaluate()
        losses = [r for r in trainer.state.log_history if "loss" in r]
        self.training = {
            "seconds": round(train_seconds, 3), "steps": trainer.state.global_step,
            "epoch": trainer.state.epoch, "trainable_parameters": trainable, "total_parameters": total,
            "trainable_percent": 100*trainable/total, "train_mean_loss": outcome.training_loss,
            "final_logged_loss": losses[-1]["loss"] if losses else None,
            "validation": validation, "peak_allocated_gpu_gb": self.torch.cuda.max_memory_allocated()/1e9,
            "log_history": trainer.state.log_history,
        }
        trainer.model.save_pretrained(OUT / "adapter")
        self.model = trainer.model
        self.model.config.use_cache = True
        self.model.eval()
        write_json(OUT / "training.json", self.training)
        print(json.dumps({k: v for k, v in self.training.items() if k != "log_history"}, ensure_ascii=False, indent=2))
        return self.training

    def save_results(self) -> dict:
        methods = ["base_closedbook", "search_only", "base_rag", "sft_closedbook"]
        loaded = {m: json.loads((OUT / f"eval_{m}.json").read_text(encoding="utf-8")) for m in methods}
        assert all(r["complete"] for r in loaded.values())
        base = loaded["base_closedbook"]["rows"]
        after = loaded["sft_closedbook"]["rows"]
        assert [r["id"] for r in base] == [r["id"] for r in after]
        regressed = [b["id"] for b, a in zip(base, after) if b["keyword_content_ok"] and not a["keyword_content_ok"]]
        improved = [b["id"] for b, a in zip(base, after) if not b["keyword_content_ok"] and a["keyword_content_ok"]]
        result = {"config": CONFIG, "environment": self.environment,
                  "dataset": json.loads((OUT / "dataset_manifest.json").read_text(encoding="utf-8")),
                  "executed_training_myself": True, "execution_note": "受講者のColab Pro T4上でAIコーディング支援を使用して実行。講師の参考結果は使用していない。",
                  "training": self.training, "summaries": {m: loaded[m]["summary"] for m in methods},
                  "regressed": regressed, "improved": improved,
                  "metric_note": "keyword_policy_rateは固定概念語と確認要求の簡易採点。意味的正しさの完全な判定ではない。"}
        write_json(OUT / "result.json", result)
        with (OUT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "category", "question", "reference"] + methods)
            w.writeheader()
            for i, gold in enumerate(self.gold):
                w.writerow({"id": gold["id"], "category": gold["category"], "question": gold["question"], "reference": gold["reference"],
                            **{m: loaded[m]["rows"][i]["answer"] for m in methods}})
        print(json.dumps({m: loaded[m]["summary"] for m in methods}, ensure_ascii=False, indent=2))
        print("regressed:", regressed, "improved:", improved)
        return result
