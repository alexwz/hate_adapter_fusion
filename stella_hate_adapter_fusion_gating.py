#!/usr/bin/env python3
"""
Leakage-safe EN->PL transfer via gated logit fusion.

This script evaluates three methods on the frozen Polish test tuples file:
1) Gated logit fusion (gate trained only on non-frozen data)
2) Direct EN adapter on Polish
3) Direct PL adapter baseline

Protocol:
- Build gate targets on non-frozen data: EN correct and PL wrong.
- Fit the gate on non-frozen train split only.
- Tune the final decision threshold on non-frozen validation by Macro F1.
- Evaluate once on frozen test and report bootstrap 95% CI half-widths.
"""

import argparse
import gc
import importlib.util
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	f1_score,
	matthews_corrcoef,
	precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	BitsAndBytesConfig,
	DataCollatorWithPadding,
	Trainer,
	TrainingArguments,
)


BASE_MODEL_ID = "sdadas/stella-pl"
MAX_LENGTH = 256
SEED = 42

METRIC_ORDER = [
	("accuracy", "Accuracy"),
	("f1_macro", "Macro F1"),
	("f1_weighted", "Weighted F1"),
	("precision_hate", "Precision (hate)"),
	("recall_hate", "Recall (hate)"),
	("f1_hate", "F1 (hate)"),
	("pr_auc", "PR AUC"),
	("mcc", "MCC"),
]


@dataclass
class EvalResult:
	method: str
	run_name: str
	key_parameter: str
	key_value: str
	metrics: Dict[str, float]
	ci_halfwidth: Dict[str, float]
	output_dir: str


def parse_args():
	p = argparse.ArgumentParser(description="Run leakage-safe gated fusion on frozen Polish test set.")
	p.add_argument("--en_adapter_dir", required=True, help="Path to EN adapter directory.")
	p.add_argument("--pl_adapter_dir", required=True, help="Path to PL adapter directory.")
	p.add_argument(
		"--frozen_test_path",
		default="frozen_test_strict_tuples.jsonl",
		help="Path to frozen tuples JSONL ([label, text, id]).",
	)
	p.add_argument("--output_dir", default="fusion_results_v2", help="Output directory.")
	p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
	p.add_argument("--batch_size", type=int, default=16)
	p.add_argument("--bootstrap_samples", type=int, default=1000)
	p.add_argument("--gate_val_size", type=float, default=0.15, help="Validation split from non-frozen set.")
	p.add_argument(
		"--gate_c_grid",
		default="0.1,1.0,10.0",
		help="Comma-separated LogisticRegression C values.",
	)
	p.add_argument(
		"--gate_scale_grid",
		default="0.5,1.0,1.5",
		help="Comma-separated multiplicative scales for gate probabilities.",
	)
	p.add_argument(
		"--threshold_grid",
		default="0.05,0.95,0.01",
		help="start,end,step for probability threshold sweep tuned by Macro F1.",
	)
	return p.parse_args()


def parse_float_grid(csv_values: str) -> List[float]:
	vals = [float(x.strip()) for x in csv_values.split(",") if x.strip()]
	if not vals:
		raise ValueError("Grid is empty")
	return vals


def parse_threshold_grid(spec: str) -> np.ndarray:
	parts = parse_float_grid(spec)
	if len(parts) != 3:
		raise ValueError("threshold_grid must be start,end,step")
	start, end, step = parts
	if step <= 0:
		raise ValueError("threshold step must be > 0")
	grid = np.arange(start, end + 1e-9, step, dtype=np.float64)
	return np.clip(grid, 0.0, 1.0)


def get_precision_config() -> Dict[str, object]:
	if not torch.cuda.is_available():
		return {
			"torch_dtype": torch.float32,
			"bnb_compute_dtype": torch.float32,
			"bf16": False,
			"fp16": False,
		}

	major, _ = torch.cuda.get_device_capability(0)
	use_bf16 = major >= 8
	return {
		"torch_dtype": torch.bfloat16 if use_bf16 else torch.float16,
		"bnb_compute_dtype": torch.bfloat16 if use_bf16 else torch.float16,
		"bf16": use_bf16,
		"fp16": not use_bf16,
	}


PRECISION_CONFIG = get_precision_config()


def cleanup_memory():
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


def find_file_by_name(root: str, target_name: str) -> str:
	for dirpath, _, filenames in os.walk(root):
		if target_name in filenames:
			return os.path.join(dirpath, target_name)
	raise FileNotFoundError(f"Could not find {target_name} under {root}")


def load_frozen_tuples(jsonl_path: str) -> pd.DataFrame:
	rows = []
	with open(jsonl_path, "r", encoding="utf-8") as f:
		for line in f:
			tup = json.loads(line)
			if not isinstance(tup, list) or len(tup) < 3:
				continue
			rows.append({"label": int(tup[0]), "text": str(tup[1]), "id": str(tup[2])})
	if not rows:
		raise RuntimeError(f"No valid tuples parsed from {jsonl_path}")
	return pd.DataFrame(rows)


def clean_html_tags(text: str) -> str:
	if not isinstance(text, str):
		return ""
	text = re.sub(r"<[^>]+>", "", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def strict_map(value):
	try:
		v = int(value)
	except (TypeError, ValueError):
		return None
	return 0 if v == 0 else 1 if v in (1, 2, 3, 4) else None


def load_non_frozen_polish_data(frozen_ids: set, hf_token=None) -> pd.DataFrame:
	kwargs = {"token": hf_token} if hf_token else {}
	ds = load_dataset("community-datasets/hate_speech_pl", **kwargs)
	full = pd.concat([split.to_pandas() for split in ds.values()], ignore_index=True)

	if "id" in full.columns:
		id_col = "id"
	elif "text_id" in full.columns:
		id_col = "text_id"
	else:
		full["row_id"] = np.arange(len(full))
		id_col = "row_id"

	full["mapped_label"] = full["rating"].map(strict_map)
	full = full.dropna(subset=["mapped_label", "text"]).copy()
	full["label"] = full["mapped_label"].astype(int)
	full["text"] = full["text"].map(clean_html_tags)
	full["id"] = full[id_col].astype(str)

	non_frozen = full[~full["id"].isin(frozen_ids)].copy()
	if len(non_frozen) < 200:
		raise RuntimeError("Too few non-frozen rows for gated fusion after ID exclusion")
	return non_frozen[["id", "text", "label"]]


def build_bnb_config():
	return BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_use_double_quant=True,
		bnb_4bit_compute_dtype=PRECISION_CONFIG["bnb_compute_dtype"],
	)


def load_model(adapter_dir: str, hf_token=None):
	has_flash = importlib.util.find_spec("flash_attn") is not None
	attn_impl = "flash_attention_2" if has_flash else "sdpa"
	hf_kwargs = {"token": hf_token} if hf_token else {}

	tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True, **hf_kwargs)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	base = AutoModelForSequenceClassification.from_pretrained(
		BASE_MODEL_ID,
		trust_remote_code=True,
		num_labels=2,
		quantization_config=build_bnb_config(),
		torch_dtype=PRECISION_CONFIG["torch_dtype"],
		attn_implementation=attn_impl,
		device_map="auto",
		**hf_kwargs,
	)
	base.config.pad_token_id = tokenizer.pad_token_id
	base.config.problem_type = "single_label_classification"

	model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
	model.eval()
	return model, tokenizer


def softmax_logits(logits: np.ndarray) -> np.ndarray:
	shifted = logits - np.max(logits, axis=-1, keepdims=True)
	expv = np.exp(shifted)
	return expv / np.sum(expv, axis=-1, keepdims=True)


def evaluate_model(model, tokenizer, df: pd.DataFrame, batch_size: int) -> Dict[str, np.ndarray]:
	texts = df["text"].tolist()
	labels = df["label"].astype(int).to_numpy()
	ids = df["id"].astype(str).to_numpy()

	collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
	hf_ds = Dataset.from_dict({"text": texts, "label": labels})

	def preprocess(batch):
		return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

	hf_ds = hf_ds.map(preprocess, batched=True)
	hf_ds = hf_ds.remove_columns(["text"])
	hf_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

	args = TrainingArguments(
		output_dir="tmp_eval",
		per_device_eval_batch_size=batch_size,
		report_to="none",
		bf16=PRECISION_CONFIG["bf16"],
		fp16=PRECISION_CONFIG["fp16"],
	)
	trainer = Trainer(model=model, args=args, tokenizer=tokenizer, data_collator=collator)
	pred_output = trainer.predict(hf_ds)
	logits = pred_output.predictions[0] if isinstance(pred_output.predictions, tuple) else pred_output.predictions

	probs = softmax_logits(logits)
	preds = np.argmax(logits, axis=-1)
	return {
		"ids": ids,
		"labels": labels,
		"logits": logits,
		"probs": probs,
		"preds": preds,
	}


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
	precision_h, recall_h, f1_h, _ = precision_recall_fscore_support(
		labels, preds, average="binary", pos_label=1, zero_division=0
	)
	return {
		"accuracy": float(accuracy_score(labels, preds)),
		"f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
		"f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
		"precision_hate": float(precision_h),
		"recall_hate": float(recall_h),
		"f1_hate": float(f1_h),
		"pr_auc": float(average_precision_score(labels, probs[:, 1])),
		"mcc": float(matthews_corrcoef(labels, preds)),
	}


def compute_preds_from_logits(logits: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
	probs = softmax_logits(logits)
	preds = (probs[:, 1] >= threshold).astype(int)
	return preds, probs


def bootstrap_ci(
	logits: np.ndarray,
	labels: np.ndarray,
	threshold: float,
	samples: int = 1000,
	seed: int = 123,
) -> Dict[str, float]:
	rng = np.random.default_rng(seed)
	n = len(labels)
	records = []
	for _ in range(samples):
		idx = rng.integers(0, n, size=n)
		sample_logits = logits[idx]
		sample_labels = labels[idx]
		sample_preds, sample_probs = compute_preds_from_logits(sample_logits, threshold)
		records.append(compute_metrics(sample_labels, sample_preds, sample_probs))

	df = pd.DataFrame(records)
	half = {}
	for m, _ in METRIC_ORDER:
		lower = float(df[m].quantile(0.025))
		upper = float(df[m].quantile(0.975))
		half[m] = (upper - lower) / 2.0
	return half


def save_prediction_artifacts(
	out_dir: str,
	ids: np.ndarray,
	labels: np.ndarray,
	logits: np.ndarray,
	threshold: float,
	extra_cols: Dict[str, np.ndarray] = None,
):
	os.makedirs(out_dir, exist_ok=True)
	preds, probs = compute_preds_from_logits(logits, threshold)
	pred_df = pd.DataFrame(
		{
			"id": ids,
			"label": labels,
			"pred": preds,
			"prob_non_hate": probs[:, 0],
			"prob_hate": probs[:, 1],
			"logit_non_hate": logits[:, 0],
			"logit_hate": logits[:, 1],
		}
	)
	if extra_cols:
		for key, val in extra_cols.items():
			pred_df[key] = val

	pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
	npz_payload = {
		"ids": ids,
		"labels": labels,
		"preds": preds,
		"probs": probs,
		"logits": logits,
		"threshold": np.array([threshold], dtype=np.float32),
	}
	if extra_cols:
		for key, val in extra_cols.items():
			npz_payload[key] = np.asarray(val)
	np.savez_compressed(os.path.join(out_dir, "predictions_arrays.npz"), **npz_payload)


def evaluate_logits(
	method: str,
	run_name: str,
	key_parameter: str,
	key_value: str,
	ids: np.ndarray,
	labels: np.ndarray,
	logits: np.ndarray,
	threshold: float,
	bootstrap_samples: int,
	save_preds_dir: str = None,
	extra_cols: Dict[str, np.ndarray] = None,
) -> EvalResult:
	preds, probs = compute_preds_from_logits(logits, threshold)
	metrics = compute_metrics(labels, preds, probs)
	ci = bootstrap_ci(logits, labels, threshold=threshold, samples=bootstrap_samples)

	if save_preds_dir is not None:
		save_prediction_artifacts(save_preds_dir, ids, labels, logits, threshold, extra_cols=extra_cols)

	return EvalResult(
		method=method,
		run_name=run_name,
		key_parameter=key_parameter,
		key_value=key_value,
		metrics=metrics,
		ci_halfwidth=ci,
		output_dir=save_preds_dir or "",
	)


def evaluate_adapter_dir(
	method: str,
	run_name: str,
	key_parameter: str,
	key_value: str,
	adapter_dir: str,
	df: pd.DataFrame,
	batch_size: int,
	bootstrap_samples: int,
	hf_token=None,
	save_preds_dir: str = None,
	threshold: float = 0.5,
) -> EvalResult:
	model, tokenizer = load_model(adapter_dir, hf_token=hf_token)
	eval_pack = evaluate_model(model, tokenizer, df, batch_size)

	del model
	cleanup_memory()

	return evaluate_logits(
		method=method,
		run_name=run_name,
		key_parameter=key_parameter,
		key_value=key_value,
		ids=eval_pack["ids"],
		labels=eval_pack["labels"],
		logits=eval_pack["logits"],
		threshold=threshold,
		bootstrap_samples=bootstrap_samples,
		save_preds_dir=save_preds_dir,
	)


def format_metric_with_ci(val: float, ci: float) -> str:
	return f"{val:.3f} $\\pm$ {ci:.3f}"


def results_to_dataframe(results: List[EvalResult]) -> pd.DataFrame:
	rows = []
	for r in results:
		row = {
			"method": r.method,
			"run_name": r.run_name,
			"key_parameter": r.key_parameter,
			"key_value": r.key_value,
		}
		for m, _ in METRIC_ORDER:
			row[m] = r.metrics[m]
			row[f"{m}_ci"] = r.ci_halfwidth[m]
		row["output_dir"] = r.output_dir
		rows.append(row)
	return pd.DataFrame(rows)


def export_supplementary_table(df: pd.DataFrame, out_path_base: str):
	df.to_csv(f"{out_path_base}.csv", index=False)

	latex_df = df[["run_name", "key_parameter", "key_value"]].copy()
	for m, label in METRIC_ORDER:
		latex_df[label] = [format_metric_with_ci(v, c) for v, c in zip(df[m], df[f"{m}_ci"])]
	latex = latex_df.to_latex(index=False, escape=False)
	with open(f"{out_path_base}.tex", "w", encoding="utf-8") as f:
		f.write(latex)


def export_summary_table(best_results: List[EvalResult], out_dir: str):
	df = results_to_dataframe(best_results)
	df.to_csv(os.path.join(out_dir, "summary_best_methods.csv"), index=False)

	latex_df = pd.DataFrame(
		{
			"Method": df["method"],
			"Key parameter": df["key_parameter"],
			"Best value": df["key_value"],
		}
	)
	for m, label in METRIC_ORDER:
		latex_df[label] = [format_metric_with_ci(v, c) for v, c in zip(df[m], df[f"{m}_ci"])]

	latex = latex_df.to_latex(index=False, escape=False)
	with open(os.path.join(out_dir, "table_summary_best_methods.tex"), "w", encoding="utf-8") as f:
		f.write(latex)


def choose_best(df: pd.DataFrame) -> pd.Series:
	return df.sort_values(["f1_macro", "f1_hate", "pr_auc"], ascending=False).iloc[0]


def build_gate_features(pl_logits: np.ndarray, en_logits: np.ndarray) -> np.ndarray:
	pl_probs = softmax_logits(pl_logits)
	en_probs = softmax_logits(en_logits)
	pl_margin = np.abs(pl_logits[:, 1] - pl_logits[:, 0])
	en_margin = np.abs(en_logits[:, 1] - en_logits[:, 0])
	pl_pred = np.argmax(pl_logits, axis=-1)
	en_pred = np.argmax(en_logits, axis=-1)
	disagree = (pl_pred != en_pred).astype(np.float32)

	features = np.stack(
		[
			pl_probs[:, 1],
			en_probs[:, 1],
			en_probs[:, 1] - pl_probs[:, 1],
			pl_margin,
			en_margin,
			en_margin - pl_margin,
			disagree,
		],
		axis=1,
	)
	return features


def build_gate_targets(pl_logits: np.ndarray, en_logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
	pl_pred = np.argmax(pl_logits, axis=-1)
	en_pred = np.argmax(en_logits, axis=-1)
	pl_correct = pl_pred == labels
	en_correct = en_pred == labels
	return (en_correct & (~pl_correct)).astype(int)


def fit_gate_model(x_train: np.ndarray, y_train: np.ndarray, c_value: float) -> Pipeline:
	# The gate is intentionally simple and calibrated with regularization.
	pipe = Pipeline(
		steps=[
			("scaler", StandardScaler()),
			(
				"clf",
				LogisticRegression(
					C=c_value,
					class_weight="balanced",
					max_iter=2000,
					random_state=SEED,
				),
			),
		]
	)
	pipe.fit(x_train, y_train)
	return pipe


def tune_threshold_for_macro_f1(logits: np.ndarray, labels: np.ndarray, threshold_grid: np.ndarray) -> Tuple[float, Dict[str, float]]:
	best_thr = float(threshold_grid[0])
	best_metrics = None
	best_macro = -1.0
	for thr in threshold_grid:
		preds, probs = compute_preds_from_logits(logits, float(thr))
		metrics = compute_metrics(labels, preds, probs)
		if metrics["f1_macro"] > best_macro:
			best_macro = metrics["f1_macro"]
			best_thr = float(thr)
			best_metrics = metrics
	return best_thr, best_metrics


def run_gated_logit_fusion(
	train_df: pd.DataFrame,
	tune_df: pd.DataFrame,
	frozen_df: pd.DataFrame,
	args,
	threshold_grid: np.ndarray,
	c_grid: List[float],
	scale_grid: List[float],
) -> Tuple[pd.DataFrame, EvalResult]:
	method = "Gated logit fusion"

	# Collect PL/EN logits once per split.
	pl_model, tokenizer = load_model(args.pl_adapter_dir, hf_token=args.hf_token)
	en_model, _ = load_model(args.en_adapter_dir, hf_token=args.hf_token)

	train_pl = evaluate_model(pl_model, tokenizer, train_df, args.batch_size)
	train_en = evaluate_model(en_model, tokenizer, train_df, args.batch_size)
	tune_pl = evaluate_model(pl_model, tokenizer, tune_df, args.batch_size)
	tune_en = evaluate_model(en_model, tokenizer, tune_df, args.batch_size)
	frozen_pl = evaluate_model(pl_model, tokenizer, frozen_df, args.batch_size)
	frozen_en = evaluate_model(en_model, tokenizer, frozen_df, args.batch_size)

	del pl_model
	del en_model
	cleanup_memory()

	if not np.array_equal(train_pl["ids"], train_en["ids"]):
		raise RuntimeError("ID mismatch between PL and EN predictions on train split")
	if not np.array_equal(tune_pl["ids"], tune_en["ids"]):
		raise RuntimeError("ID mismatch between PL and EN predictions on tune split")
	if not np.array_equal(frozen_pl["ids"], frozen_en["ids"]):
		raise RuntimeError("ID mismatch between PL and EN predictions on frozen split")

	x_train = build_gate_features(train_pl["logits"], train_en["logits"])
	y_train = build_gate_targets(train_pl["logits"], train_en["logits"], train_df["label"].to_numpy())
	x_tune = build_gate_features(tune_pl["logits"], tune_en["logits"])
	x_frozen = build_gate_features(frozen_pl["logits"], frozen_en["logits"])

	records: List[EvalResult] = []
	models_by_c: Dict[float, Pipeline] = {}
	best_cfg = None
	best_score = -1.0

	for c_val in c_grid:
		gate = fit_gate_model(x_train, y_train, c_val)
		models_by_c[c_val] = gate
		g_tune = gate.predict_proba(x_tune)[:, 1]

		for scale in scale_grid:
			g_scaled = np.clip(scale * g_tune, 0.0, 1.0)
			fused_logits_tune = (1.0 - g_scaled[:, None]) * tune_pl["logits"] + g_scaled[:, None] * tune_en["logits"]
			best_thr, _ = tune_threshold_for_macro_f1(
				fused_logits_tune,
				tune_df["label"].to_numpy(),
				threshold_grid,
			)
			run_name = f"gate_C{c_val:.3g}_scale{scale:.2f}"
			result = evaluate_logits(
				method=method,
				run_name=run_name,
				key_parameter="gate_C,gate_scale,threshold",
				key_value=f"{c_val:.3g},{scale:.2f},{best_thr:.2f}",
				ids=tune_pl["ids"],
				labels=tune_df["label"].to_numpy(),
				logits=fused_logits_tune,
				threshold=best_thr,
				bootstrap_samples=args.bootstrap_samples,
				save_preds_dir=None,
			)
			records.append(result)

			score = result.metrics["f1_macro"]
			if score > best_score:
				best_score = score
				best_cfg = (c_val, scale, best_thr, run_name)

	if best_cfg is None:
		raise RuntimeError("No gated fusion config evaluated")

	best_c, best_scale, best_thr, best_run_name = best_cfg
	best_gate = models_by_c[best_c]
	g_frozen = best_gate.predict_proba(x_frozen)[:, 1]
	g_frozen_scaled = np.clip(best_scale * g_frozen, 0.0, 1.0)
	fused_logits_frozen = (
		(1.0 - g_frozen_scaled[:, None]) * frozen_pl["logits"]
		+ g_frozen_scaled[:, None] * frozen_en["logits"]
	)

	pred_dir = os.path.join(args.output_dir, "best_runs", method.replace(" ", "_"))
	extra_cols = {
		"gate_prob": g_frozen,
		"gate_prob_scaled": g_frozen_scaled,
		"pl_prob_hate": softmax_logits(frozen_pl["logits"])[:, 1],
		"en_prob_hate": softmax_logits(frozen_en["logits"])[:, 1],
	}
	best_result = evaluate_logits(
		method=method,
		run_name=best_run_name,
		key_parameter="gate_C,gate_scale,threshold",
		key_value=f"{best_c:.3g},{best_scale:.2f},{best_thr:.2f}",
		ids=frozen_pl["ids"],
		labels=frozen_df["label"].to_numpy(),
		logits=fused_logits_frozen,
		threshold=best_thr,
		bootstrap_samples=args.bootstrap_samples,
		save_preds_dir=pred_dir,
		extra_cols=extra_cols,
	)

	gate_summary = {
		"best_c": float(best_c),
		"best_scale": float(best_scale),
		"best_threshold": float(best_thr),
		"train_size": int(len(train_df)),
		"tune_size": int(len(tune_df)),
	}
	os.makedirs(pred_dir, exist_ok=True)
	with open(os.path.join(pred_dir, "gate_config.json"), "w", encoding="utf-8") as f:
		json.dump(gate_summary, f, indent=2)

	sweep_df = results_to_dataframe(records)
	return sweep_df, best_result


def run_direct_en(frozen_df: pd.DataFrame, args) -> Tuple[pd.DataFrame, EvalResult]:
	method = "Direct EN adapter on Polish"
	run_name = "direct_en"
	pred_dir = os.path.join(args.output_dir, "best_runs", method.replace(" ", "_"))
	result = evaluate_adapter_dir(
		method=method,
		run_name=run_name,
		key_parameter="threshold",
		key_value="0.50",
		adapter_dir=args.en_adapter_dir,
		df=frozen_df,
		batch_size=args.batch_size,
		bootstrap_samples=args.bootstrap_samples,
		hf_token=args.hf_token,
		save_preds_dir=pred_dir,
		threshold=0.5,
	)
	return results_to_dataframe([result]), result


def run_direct_pl(frozen_df: pd.DataFrame, args) -> Tuple[pd.DataFrame, EvalResult]:
	method = "Direct PL adapter"
	run_name = "direct_pl"
	pred_dir = os.path.join(args.output_dir, "best_runs", method.replace(" ", "_"))
	result = evaluate_adapter_dir(
		method=method,
		run_name=run_name,
		key_parameter="threshold",
		key_value="0.50",
		adapter_dir=args.pl_adapter_dir,
		df=frozen_df,
		batch_size=args.batch_size,
		bootstrap_samples=args.bootstrap_samples,
		hf_token=args.hf_token,
		save_preds_dir=pred_dir,
		threshold=0.5,
	)
	return results_to_dataframe([result]), result


def main():
	args = parse_args()
	os.makedirs(args.output_dir, exist_ok=True)

	print(
		f"Precision config: torch_dtype={PRECISION_CONFIG['torch_dtype']}, "
		f"bnb_compute_dtype={PRECISION_CONFIG['bnb_compute_dtype']}, "
		f"bf16={PRECISION_CONFIG['bf16']}, fp16={PRECISION_CONFIG['fp16']}"
	)

	cwd = os.getcwd()
	frozen_path = args.frozen_test_path
	if not os.path.isabs(frozen_path):
		if os.path.exists(frozen_path):
			frozen_path = os.path.abspath(frozen_path)
		else:
			frozen_path = find_file_by_name(cwd, os.path.basename(frozen_path))

	print(f"Using frozen test file: {frozen_path}")
	frozen_df = load_frozen_tuples(frozen_path)
	print(f"Frozen examples: {len(frozen_df)}")

	non_frozen_df = load_non_frozen_polish_data(set(frozen_df["id"].astype(str).tolist()), hf_token=args.hf_token)
	train_df, tune_df = train_test_split(
		non_frozen_df,
		test_size=args.gate_val_size,
		random_state=SEED,
		stratify=non_frozen_df["label"],
	)
	print(f"Gate train examples: {len(train_df)}")
	print(f"Gate tuning examples: {len(tune_df)}")

	c_grid = parse_float_grid(args.gate_c_grid)
	scale_grid = parse_float_grid(args.gate_scale_grid)
	threshold_grid = parse_threshold_grid(args.threshold_grid)

	best_results: List[EvalResult] = []

	sweep_df, best = run_gated_logit_fusion(
		train_df=train_df,
		tune_df=tune_df,
		frozen_df=frozen_df,
		args=args,
		threshold_grid=threshold_grid,
		c_grid=c_grid,
		scale_grid=scale_grid,
	)
	export_supplementary_table(sweep_df, os.path.join(args.output_dir, "appendix_gated_logit_fusion_sweep"))
	best_results.append(best)

	sweep_df, best = run_direct_en(frozen_df, args)
	export_supplementary_table(sweep_df, os.path.join(args.output_dir, "appendix_direct_en"))
	best_results.append(best)

	sweep_df, best = run_direct_pl(frozen_df, args)
	export_supplementary_table(sweep_df, os.path.join(args.output_dir, "appendix_direct_pl"))
	best_results.append(best)

	export_summary_table(best_results, args.output_dir)
	print("Done. Summary + supplementary tables exported.")


if __name__ == "__main__":
	main()
