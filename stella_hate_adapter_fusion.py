#!/usr/bin/env python3
"""
Adapter fusion experiments for Polish hate-speech detection.

Implements and evaluates five methods on the frozen Polish test tuples file:
1) Layer-weighted task-vector scaling
2) Layer-selective interpolation
3) Head-only transfer
4) Knowledge distillation
5) Direct EN adapter on Polish (no merging)

For each method, the script runs a hyperparameter sweep, reports full sweep tables,
selects the best run by macro F1, and compiles one summary table across methods.
All reported metrics include bootstrap 95% CI half-widths (± delta) from 1,000
resamples by default.
"""

import argparse
import gc
import importlib.util
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	f1_score,
	matthews_corrcoef,
	precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
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

# Stage-1 cosines from prior probing run (default for adaptive layer weighting).
DEFAULT_STAGE1_COSINES = [
	0.0001, -0.0866, -0.1405, -0.0835, -0.0072, -0.0490, -0.0065, 0.0052,
	-0.0441, -0.0240, -0.0614, -0.0141, -0.0236, 0.1151, 0.1726, 0.2139,
	0.2849, 0.3232, 0.2924, 0.3255, 0.3254, 0.3059, 0.2772, 0.3403,
	0.3955, 0.4273, 0.4552, 0.4126,
]

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
	p = argparse.ArgumentParser(description="Run adapter fusion sweeps on frozen Polish test set.")
	p.add_argument("--en_adapter_dir", required=True, help="Path to EN adapter directory.")
	p.add_argument("--pl_adapter_dir", required=True, help="Path to PL adapter directory.")
	p.add_argument(
		"--frozen_test_path",
		default="frozen_test_strict_tuples.jsonl",
		help="Path to frozen tuples JSONL ([label, text, id]).",
	)
	p.add_argument("--output_dir", default="fusion_results", help="Output directory.")
	p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
	p.add_argument("--batch_size", type=int, default=16)
	p.add_argument("--bootstrap_samples", type=int, default=1000)
	p.add_argument(
		"--stage1_cosines",
		default=",".join(str(v) for v in DEFAULT_STAGE1_COSINES),
		help="Comma-separated per-layer cosine values from Stage 1.",
	)
	p.add_argument("--run_distillation", action="store_true", help="Enable distillation sweep.")
	return p.parse_args()


def get_precision_config() -> Dict[str, object]:
	"""Return GPU-appropriate precision settings.

	T4 does not support bf16 efficiently, so use fp16 there. On Ampere/Hopper,
	bfloat16 is preferred when available.
	"""
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


def parse_stage1_cosines(value: str) -> List[float]:
	vals = [float(x.strip()) for x in value.split(",") if x.strip()]
	if not vals:
		raise ValueError("stage1_cosines is empty")
	return vals


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
		raise RuntimeError("Too few non-frozen rows for distillation after ID exclusion")
	return non_frozen[["id", "text", "label"]]


def build_bnb_config():
	return BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_use_double_quant=True,
		bnb_4bit_compute_dtype=PRECISION_CONFIG["bnb_compute_dtype"],
	)


def load_model(adapter_dir: str, hf_token=None, is_trainable: bool = False):
	has_flash = importlib.util.find_spec("flash_attn") is not None
	attn_impl = "flash_attention_2" if has_flash else "sdpa"
	hf_kwargs = {"token": hf_token} if hf_token else {}

	# Fused temporary adapter directories only contain adapter files, not the
	# custom tokenizer implementation from the base model repo.
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
	if is_trainable:
		base = prepare_model_for_kbit_training(base)

	model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=is_trainable)
	if is_trainable:
		for param in model.parameters():
			if param.requires_grad and param.dtype != torch.float32:
				param.data = param.data.float()
	model.eval()
	return model, tokenizer


def load_adapter_state(adapter_dir: str) -> Tuple[Dict[str, torch.Tensor], str]:
	safe_path = os.path.join(adapter_dir, "adapter_model.safetensors")
	bin_path = os.path.join(adapter_dir, "adapter_model.bin")
	if os.path.exists(safe_path):
		return load_safetensors(safe_path), "safetensors"
	if os.path.exists(bin_path):
		return torch.load(bin_path, map_location="cpu"), "bin"
	raise FileNotFoundError(f"No adapter model file found in {adapter_dir}")


def save_adapter_state(template_adapter_dir: str, state: Dict[str, torch.Tensor], out_dir: str):
	os.makedirs(out_dir, exist_ok=True)
	for fname in ["adapter_config.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"]:
		src = os.path.join(template_adapter_dir, fname)
		if os.path.exists(src):
			shutil.copy2(src, os.path.join(out_dir, fname))
	save_safetensors(state, os.path.join(out_dir, "adapter_model.safetensors"))


def parse_layer_from_key(key: str):
	m = re.search(r"\.layers\.(\d+)\.", key)
	return int(m.group(1)) if m else None


def is_lora_key(key: str) -> bool:
	return ".lora_A." in key or ".lora_B." in key


def is_head_key(key: str) -> bool:
	return "modules_to_save" in key and (".score." in key or ".classifier." in key)


def blend(a: torch.Tensor, b: torch.Tensor, coeff_b: float) -> torch.Tensor:
	return (1.0 - coeff_b) * a + coeff_b * b


def task_vector_scaled_state(
	state_pl: Dict[str, torch.Tensor],
	state_en: Dict[str, torch.Tensor],
	alpha_base: float,
	stage1_cosines: List[float],
	head_beta: float,
) -> Dict[str, torch.Tensor]:
	out = {}
	for key, w_pl in state_pl.items():
		w_en = state_en.get(key)
		if w_en is None:
			out[key] = w_pl.clone()
			continue
		if is_lora_key(key):
			layer = parse_layer_from_key(key)
			if layer is None or layer >= len(stage1_cosines):
				coeff = alpha_base
			else:
				penalty = max(0.0, 1.0 - abs(stage1_cosines[layer]) / 0.5)
				coeff = alpha_base * penalty
			out[key] = w_pl + coeff * w_en
		elif is_head_key(key):
			out[key] = blend(w_pl, w_en, head_beta)
		else:
			out[key] = w_pl.clone()
	return out


def layer_selective_state(
	state_pl: Dict[str, torch.Tensor],
	state_en: Dict[str, torch.Tensor],
	lambda_mid: float,
	lambda_upper: float,
	head_beta: float,
) -> Dict[str, torch.Tensor]:
	out = {}
	for key, w_pl in state_pl.items():
		w_en = state_en.get(key)
		if w_en is None:
			out[key] = w_pl.clone()
			continue
		if is_lora_key(key):
			layer = parse_layer_from_key(key)
			coeff = 0.0
			if layer is not None:
				if 11 <= layer <= 14:
					coeff = lambda_mid
				elif 15 <= layer <= 22:
					coeff = lambda_upper
			out[key] = blend(w_pl, w_en, coeff)
		elif is_head_key(key):
			out[key] = blend(w_pl, w_en, head_beta)
		else:
			out[key] = w_pl.clone()
	return out


def head_only_state(
	state_pl: Dict[str, torch.Tensor],
	state_en: Dict[str, torch.Tensor],
	head_beta: float,
) -> Dict[str, torch.Tensor]:
	out = {}
	for key, w_pl in state_pl.items():
		w_en = state_en.get(key)
		if w_en is None:
			out[key] = w_pl.clone()
			continue
		if is_head_key(key):
			out[key] = blend(w_pl, w_en, head_beta)
		else:
			out[key] = w_pl.clone()
	return out


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

	shifted = logits - np.max(logits, axis=-1, keepdims=True)
	expv = np.exp(shifted)
	probs = expv / np.sum(expv, axis=-1, keepdims=True)
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


def bootstrap_ci(logits: np.ndarray, labels: np.ndarray, samples: int = 1000, seed: int = 123) -> Dict[str, float]:
	rng = np.random.default_rng(seed)
	n = len(labels)
	records = []
	for _ in range(samples):
		idx = rng.integers(0, n, size=n)
		sample_logits = logits[idx]
		sample_labels = labels[idx]
		sample_preds = np.argmax(sample_logits, axis=-1)
		shifted = sample_logits - np.max(sample_logits, axis=-1, keepdims=True)
		expv = np.exp(shifted)
		sample_probs = expv / np.sum(expv, axis=-1, keepdims=True)
		records.append(compute_metrics(sample_labels, sample_preds, sample_probs))

	df = pd.DataFrame(records)
	half = {}
	for m, _ in METRIC_ORDER:
		lower = float(df[m].quantile(0.025))
		upper = float(df[m].quantile(0.975))
		half[m] = (upper - lower) / 2.0
	return half


def save_prediction_artifacts(out_dir: str, eval_pack: Dict[str, np.ndarray]):
	os.makedirs(out_dir, exist_ok=True)
	pred_df = pd.DataFrame(
		{
			"id": eval_pack["ids"],
			"label": eval_pack["labels"],
			"pred": eval_pack["preds"],
			"prob_non_hate": eval_pack["probs"][:, 0],
			"prob_hate": eval_pack["probs"][:, 1],
			"logit_non_hate": eval_pack["logits"][:, 0],
			"logit_hate": eval_pack["logits"][:, 1],
		}
	)
	pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
	np.savez_compressed(
		os.path.join(out_dir, "predictions_arrays.npz"),
		ids=eval_pack["ids"],
		labels=eval_pack["labels"],
		preds=eval_pack["preds"],
		probs=eval_pack["probs"],
		logits=eval_pack["logits"],
	)


def evaluate_adapter_dir(
	method: str,
	run_name: str,
	key_parameter: str,
	key_value: str,
	adapter_dir: str,
	frozen_df: pd.DataFrame,
	batch_size: int,
	bootstrap_samples: int,
	hf_token=None,
	save_preds_dir: str = None,
) -> EvalResult:
	model, tokenizer = load_model(adapter_dir, hf_token=hf_token)
	eval_pack = evaluate_model(model, tokenizer, frozen_df, batch_size)
	metrics = compute_metrics(eval_pack["labels"], eval_pack["preds"], eval_pack["probs"])
	ci = bootstrap_ci(eval_pack["logits"], eval_pack["labels"], samples=bootstrap_samples)

	if save_preds_dir is not None:
		save_prediction_artifacts(save_preds_dir, eval_pack)

	del model
	cleanup_memory()

	return EvalResult(
		method=method,
		run_name=run_name,
		key_parameter=key_parameter,
		key_value=key_value,
		metrics=metrics,
		ci_halfwidth=ci,
		output_dir=save_preds_dir or "",
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


def run_task_vector_sweep(
	state_pl,
	state_en,
	pl_adapter_dir,
	tune_df,
	frozen_df,
	stage1_cosines,
	args,
) -> Tuple[pd.DataFrame, EvalResult]:
	method = "Layer-weighted task-vector scaling"
	records = []
	alpha_grid = [0.10, 0.15, 0.20, 0.25]
	head_grid = [0.0, 0.3]

	for alpha in alpha_grid:
		for head_beta in head_grid:
			run_name = f"taskvec_a{alpha:.2f}_hb{head_beta:.2f}"
			with tempfile.TemporaryDirectory(prefix="fused_taskvec_") as td:
				fused_state = task_vector_scaled_state(state_pl, state_en, alpha, stage1_cosines, head_beta)
				save_adapter_state(pl_adapter_dir, fused_state, td)
				out_pred_dir = os.path.join(args.output_dir, "runs", method, run_name)
				result = evaluate_adapter_dir(
					method=method,
					run_name=run_name,
					key_parameter="alpha_base,head_beta",
					key_value=f"{alpha:.2f},{head_beta:.2f}",
					adapter_dir=td,
					frozen_df=tune_df,
					batch_size=args.batch_size,
					bootstrap_samples=args.bootstrap_samples,
					hf_token=args.hf_token,
					save_preds_dir=None,
				)
				records.append(result)

	sweep_df = results_to_dataframe(records)
	best_row = choose_best(sweep_df)
	best_alpha, best_head = [float(x) for x in best_row["key_value"].split(",")]

	# Re-evaluate best run and persist predictions.
	best_run_name = best_row["run_name"]
	with tempfile.TemporaryDirectory(prefix="fused_taskvec_best_") as td:
		fused_state = task_vector_scaled_state(state_pl, state_en, best_alpha, stage1_cosines, best_head)
		save_adapter_state(pl_adapter_dir, fused_state, td)
		pred_dir = os.path.join(args.output_dir, "best_runs", method.replace(" ", "_"))
		best_result = evaluate_adapter_dir(
			method=method,
			run_name=best_run_name,
			key_parameter="alpha_base,head_beta",
			key_value=f"{best_alpha:.2f},{best_head:.2f}",
			adapter_dir=td,
			frozen_df=frozen_df,
			batch_size=args.batch_size,
			bootstrap_samples=args.bootstrap_samples,
			hf_token=args.hf_token,
			save_preds_dir=pred_dir,
		)

	return sweep_df, best_result


def run_layer_selective_sweep(
	state_pl,
	state_en,
	pl_adapter_dir,
	tune_df,
	frozen_df,
	args,
) -> Tuple[pd.DataFrame, EvalResult]:
	method = "Layer-selective interpolation"
	records = []
	lambda_mid_grid = [0.3, 0.5]
	lambda_upper_grid = [0.1, 0.2, 0.3]
	head_grid = [0.0, 0.3]

	for lm in lambda_mid_grid:
		for lu in lambda_upper_grid:
			for hb in head_grid:
				run_name = f"layerinterp_lm{lm:.2f}_lu{lu:.2f}_hb{hb:.2f}"
				with tempfile.TemporaryDirectory(prefix="fused_layerinterp_") as td:
					fused_state = layer_selective_state(state_pl, state_en, lm, lu, hb)
					save_adapter_state(pl_adapter_dir, fused_state, td)
					result = evaluate_adapter_dir(
						method=method,
						run_name=run_name,
						key_parameter="lambda_mid,lambda_upper,head_beta",
						key_value=f"{lm:.2f},{lu:.2f},{hb:.2f}",
						adapter_dir=td,
						frozen_df=tune_df,
						batch_size=args.batch_size,
						bootstrap_samples=args.bootstrap_samples,
						hf_token=args.hf_token,
						save_preds_dir=None,
					)
					records.append(result)

	sweep_df = results_to_dataframe(records)
	best_row = choose_best(sweep_df)
	lm, lu, hb = [float(x) for x in best_row["key_value"].split(",")]
	best_run_name = best_row["run_name"]

	with tempfile.TemporaryDirectory(prefix="fused_layerinterp_best_") as td:
		fused_state = layer_selective_state(state_pl, state_en, lm, lu, hb)
		save_adapter_state(pl_adapter_dir, fused_state, td)
		pred_dir = os.path.join(args.output_dir, "best_runs", method.replace(" ", "_"))
		best_result = evaluate_adapter_dir(
			method=method,
			run_name=best_run_name,
			key_parameter="lambda_mid,lambda_upper,head_beta",
			key_value=f"{lm:.2f},{lu:.2f},{hb:.2f}",
			adapter_dir=td,
			frozen_df=frozen_df,
			batch_size=args.batch_size,
			bootstrap_samples=args.bootstrap_samples,
			hf_token=args.hf_token,
			save_preds_dir=pred_dir,
		)

	return sweep_df, best_result


def run_head_only_sweep(
	state_pl,
	state_en,
	pl_adapter_dir,
	tune_df,
	frozen_df,
	args,
) -> Tuple[pd.DataFrame, EvalResult]:
	method = "Head-only transfer"
	records = []
	head_grid = [0.1, 0.2, 0.3, 0.5, 1.0]

	for hb in head_grid:
		run_name = f"head_beta_{hb:.2f}"
		with tempfile.TemporaryDirectory(prefix="fused_headonly_") as td:
			fused_state = head_only_state(state_pl, state_en, hb)
			save_adapter_state(pl_adapter_dir, fused_state, td)
			result = evaluate_adapter_dir(
				method=method,
				run_name=run_name,
				key_parameter="head_beta",
				key_value=f"{hb:.2f}",
				adapter_dir=td,
				frozen_df=tune_df,
				batch_size=args.batch_size,
				bootstrap_samples=args.bootstrap_samples,
				hf_token=args.hf_token,
				save_preds_dir=None,
			)
			records.append(result)

	sweep_df = results_to_dataframe(records)
	best_row = choose_best(sweep_df)
	hb = float(best_row["key_value"])
	best_run_name = best_row["run_name"]

	with tempfile.TemporaryDirectory(prefix="fused_headonly_best_") as td:
		fused_state = head_only_state(state_pl, state_en, hb)
		save_adapter_state(pl_adapter_dir, fused_state, td)
		pred_dir = os.path.join(args.output_dir, "best_runs", method.replace(" ", "_"))
		best_result = evaluate_adapter_dir(
			method=method,
			run_name=best_run_name,
			key_parameter="head_beta",
			key_value=f"{hb:.2f}",
			adapter_dir=td,
			frozen_df=frozen_df,
			batch_size=args.batch_size,
			bootstrap_samples=args.bootstrap_samples,
			hf_token=args.hf_token,
			save_preds_dir=pred_dir,
		)

	return sweep_df, best_result


class DistillDataset(torch.utils.data.Dataset):
	def __init__(self, encodings, labels, teacher_logits):
		self.encodings = encodings
		self.labels = labels
		self.teacher_logits = teacher_logits

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
		item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
		item["teacher_logits"] = torch.tensor(self.teacher_logits[idx], dtype=torch.float32)
		return item


class DistillTrainer(Trainer):
	def __init__(self, distill_beta: float, temperature: float, **kwargs):
		super().__init__(**kwargs)
		self.distill_beta = distill_beta
		self.temperature = temperature
		self.ce = torch.nn.CrossEntropyLoss()
		self.kl = torch.nn.KLDivLoss(reduction="batchmean")

	def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
		labels = inputs.pop("labels")
		teacher_logits = inputs.pop("teacher_logits")
		outputs = model(**inputs)
		logits = outputs.logits

		ce_loss = self.ce(logits, labels)
		t = self.temperature
		log_p = torch.nn.functional.log_softmax(logits / t, dim=-1)
		teacher_probs = torch.softmax(teacher_logits / t, dim=-1)
		kl_loss = self.kl(log_p, teacher_probs)
		loss = (1.0 - self.distill_beta) * ce_loss + self.distill_beta * (t * t) * kl_loss
		return (loss, outputs) if return_outputs else loss


def predict_teacher_logits(adapter_dir: str, tokenizer, texts: List[str], batch_size: int, hf_token=None) -> np.ndarray:
	model, _ = load_model(adapter_dir, hf_token=hf_token)
	device = next(model.parameters()).device
	logits_all = []
	for i in range(0, len(texts), batch_size):
		batch = texts[i : i + batch_size]
		inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True).to(device)
		with torch.inference_mode():
			logits = model(**inputs).logits
		logits_all.append(logits.cpu().numpy())
	del model
	cleanup_memory()
	return np.concatenate(logits_all, axis=0)


def run_distillation_sweep(
	state_pl,
	state_en,
	pl_adapter_dir,
	en_adapter_dir,
	train_df,
	val_df,
	frozen_df,
	args,
) -> Tuple[pd.DataFrame, EvalResult]:
	method = "Knowledge distillation"

	_, tokenizer = load_model(pl_adapter_dir, hf_token=args.hf_token)
	train_teacher_logits = predict_teacher_logits(
		en_adapter_dir,
		tokenizer,
		train_df["text"].tolist(),
		args.batch_size,
		hf_token=args.hf_token,
	)
	val_teacher_logits = predict_teacher_logits(
		en_adapter_dir,
		tokenizer,
		val_df["text"].tolist(),
		args.batch_size,
		hf_token=args.hf_token,
	)

	def encode_texts(texts):
		return tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=True)

	train_enc = encode_texts(train_df["text"].tolist())
	val_enc = encode_texts(val_df["text"].tolist())

	records = []
	t_grid = [1.0, 2.0, 4.0]
	b_grid = [0.1, 0.2, 0.3]
	head_grid = [0.0, 0.3]
	best_cfg = None
	best_val_loss = float("inf")

	for temp in t_grid:
		for beta in b_grid:
			for head_beta in head_grid:
				run_name = f"distill_T{temp:.1f}_b{beta:.2f}_hb{head_beta:.2f}"
				with tempfile.TemporaryDirectory(prefix="fused_distill_init_") as init_td:
					init_state = head_only_state(state_pl, state_en, head_beta)
					save_adapter_state(pl_adapter_dir, init_state, init_td)

					model, _ = load_model(init_td, hf_token=args.hf_token, is_trainable=True)
					model.train()
					for p in model.parameters():
						p.requires_grad = False
					for n, p in model.named_parameters():
						if "lora_" in n or "modules_to_save" in n:
							p.requires_grad = True
							if p.dtype != torch.float32:
								p.data = p.data.float()

					train_ds = DistillDataset(train_enc, train_df["label"].to_numpy(), train_teacher_logits)
					val_ds = DistillDataset(val_enc, val_df["label"].to_numpy(), val_teacher_logits)

					train_args = TrainingArguments(
						output_dir=os.path.join(args.output_dir, "tmp_distill", run_name),
						overwrite_output_dir=True,
						num_train_epochs=2,
						learning_rate=1e-4,
						weight_decay=0.01,
						warmup_ratio=0.1,
						lr_scheduler_type="cosine",
						per_device_train_batch_size=4,
						per_device_eval_batch_size=8,
						gradient_accumulation_steps=2,
						eval_strategy="epoch",
						save_strategy="epoch",
						save_total_limit=1,
						load_best_model_at_end=True,
						metric_for_best_model="eval_loss",
						greater_is_better=False,
						report_to="none",
						remove_unused_columns=False,
						bf16=PRECISION_CONFIG["bf16"],
						fp16=PRECISION_CONFIG["fp16"],
						seed=SEED,
					)
					trainer = DistillTrainer(
						distill_beta=beta,
						temperature=temp,
						model=model,
						args=train_args,
						train_dataset=train_ds,
						eval_dataset=val_ds,
						tokenizer=tokenizer,
					)
					trainer.train()
					eval_out = trainer.evaluate()
					val_loss = float(eval_out.get("eval_loss", float("inf")))
					if val_loss < best_val_loss:
						best_val_loss = val_loss
						best_cfg = (temp, beta, head_beta, run_name)

					with tempfile.TemporaryDirectory(prefix="fused_distill_final_") as final_td:
						model.save_pretrained(final_td)
						tokenizer.save_pretrained(final_td)
						result = evaluate_adapter_dir(
							method=method,
							run_name=run_name,
							key_parameter="temperature,beta,head_beta",
							key_value=f"{temp:.1f},{beta:.2f},{head_beta:.2f}",
							adapter_dir=final_td,
							frozen_df=val_df,
							batch_size=args.batch_size,
							bootstrap_samples=args.bootstrap_samples,
							hf_token=args.hf_token,
							save_preds_dir=None,
						)
						records.append(result)

					del trainer
					del model
					cleanup_memory()

	sweep_df = results_to_dataframe(records)
	if best_cfg is not None:
		temp, beta, head_beta, best_run_name = best_cfg
	else:
		best_row = choose_best(sweep_df)
		temp, beta, head_beta = [float(x) for x in best_row["key_value"].split(",")]
		best_run_name = best_row["run_name"]

	# Re-train best setting and serialize final predictions.
	with tempfile.TemporaryDirectory(prefix="fused_distill_best_init_") as init_td:
		init_state = head_only_state(state_pl, state_en, head_beta)
		save_adapter_state(pl_adapter_dir, init_state, init_td)
		model, _ = load_model(init_td, hf_token=args.hf_token, is_trainable=True)
		model.train()
		for p in model.parameters():
			p.requires_grad = False
		for n, p in model.named_parameters():
			if "lora_" in n or "modules_to_save" in n:
				p.requires_grad = True
				if p.dtype != torch.float32:
					p.data = p.data.float()

		train_ds = DistillDataset(train_enc, train_df["label"].to_numpy(), train_teacher_logits)
		val_ds = DistillDataset(val_enc, val_df["label"].to_numpy(), val_teacher_logits)
		train_args = TrainingArguments(
			output_dir=os.path.join(args.output_dir, "tmp_distill", "best_refit"),
			overwrite_output_dir=True,
			num_train_epochs=2,
			learning_rate=1e-4,
			weight_decay=0.01,
			warmup_ratio=0.1,
			lr_scheduler_type="cosine",
			per_device_train_batch_size=4,
			per_device_eval_batch_size=8,
			gradient_accumulation_steps=2,
			eval_strategy="epoch",
			save_strategy="epoch",
			save_total_limit=1,
			load_best_model_at_end=True,
			metric_for_best_model="eval_loss",
			greater_is_better=False,
			report_to="none",
			remove_unused_columns=False,
			bf16=PRECISION_CONFIG["bf16"],
			fp16=PRECISION_CONFIG["fp16"],
			seed=SEED,
		)
		trainer = DistillTrainer(
			distill_beta=beta,
			temperature=temp,
			model=model,
			args=train_args,
			train_dataset=train_ds,
			eval_dataset=val_ds,
			tokenizer=tokenizer,
		)
		trainer.train()

		with tempfile.TemporaryDirectory(prefix="fused_distill_best_final_") as final_td:
			model.save_pretrained(final_td)
			tokenizer.save_pretrained(final_td)
			pred_dir = os.path.join(args.output_dir, "best_runs", method.replace(" ", "_"))
			best_result = evaluate_adapter_dir(
				method=method,
				run_name=best_run_name,
				key_parameter="temperature,beta,head_beta",
				key_value=f"{temp:.1f},{beta:.2f},{head_beta:.2f}",
				adapter_dir=final_td,
				frozen_df=frozen_df,
				batch_size=args.batch_size,
				bootstrap_samples=args.bootstrap_samples,
				hf_token=args.hf_token,
				save_preds_dir=pred_dir,
			)

		del trainer
		del model
		cleanup_memory()

	return sweep_df, best_result


def run_direct_en(
	en_adapter_dir,
	frozen_df,
	args,
) -> Tuple[pd.DataFrame, EvalResult]:
	method = "Direct EN adapter on Polish"
	run_name = "direct_en"
	pred_dir = os.path.join(args.output_dir, "best_runs", method.replace(" ", "_"))
	result = evaluate_adapter_dir(
		method=method,
		run_name=run_name,
		key_parameter="none",
		key_value="-",
		adapter_dir=en_adapter_dir,
		frozen_df=frozen_df,
		batch_size=args.batch_size,
		bootstrap_samples=args.bootstrap_samples,
		hf_token=args.hf_token,
		save_preds_dir=pred_dir,
	)
	df = results_to_dataframe([result])
	return df, result


def run_direct_pl(
	pl_adapter_dir,
	frozen_df,
	args,
) -> Tuple[pd.DataFrame, EvalResult]:
	method = "Direct PL adapter"
	run_name = "direct_pl"
	pred_dir = os.path.join(args.output_dir, "best_runs", method.replace(" ", "_"))
	result = evaluate_adapter_dir(
		method=method,
		run_name=run_name,
		key_parameter="none",
		key_value="-",
		adapter_dir=pl_adapter_dir,
		frozen_df=frozen_df,
		batch_size=args.batch_size,
		bootstrap_samples=args.bootstrap_samples,
		hf_token=args.hf_token,
		save_preds_dir=pred_dir,
	)
	df = results_to_dataframe([result])
	return df, result


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

	# No-leakage protocol: tune all hyperparameters on non-frozen validation only.
	non_frozen_df = load_non_frozen_polish_data(set(frozen_df["id"].astype(str).tolist()), hf_token=args.hf_token)
	train_df, tune_df = train_test_split(
		non_frozen_df,
		test_size=0.15,
		random_state=SEED,
		stratify=non_frozen_df["label"],
	)
	print(f"Tuning examples (non-frozen val): {len(tune_df)}")

	stage1_cosines = parse_stage1_cosines(args.stage1_cosines)

	state_en, _ = load_adapter_state(args.en_adapter_dir)
	state_pl, _ = load_adapter_state(args.pl_adapter_dir)

	best_results = []

	# 1) Layer-weighted task-vector scaling
	sweep_df, best = run_task_vector_sweep(state_pl, state_en, args.pl_adapter_dir, tune_df, frozen_df, stage1_cosines, args)
	export_supplementary_table(sweep_df, os.path.join(args.output_dir, "appendix_task_vector_sweep"))
	best_results.append(best)

	# 2) Layer-selective interpolation
	sweep_df, best = run_layer_selective_sweep(state_pl, state_en, args.pl_adapter_dir, tune_df, frozen_df, args)
	export_supplementary_table(sweep_df, os.path.join(args.output_dir, "appendix_layer_selective_sweep"))
	best_results.append(best)

	# 3) Head-only transfer
	sweep_df, best = run_head_only_sweep(state_pl, state_en, args.pl_adapter_dir, tune_df, frozen_df, args)
	export_supplementary_table(sweep_df, os.path.join(args.output_dir, "appendix_head_only_sweep"))
	best_results.append(best)

	# 4) Knowledge distillation
	if args.run_distillation:
		sweep_df, best = run_distillation_sweep(
			state_pl,
			state_en,
			args.pl_adapter_dir,
			args.en_adapter_dir,
			train_df,
			tune_df,
			frozen_df,
			args,
		)
		export_supplementary_table(sweep_df, os.path.join(args.output_dir, "appendix_distillation_sweep"))
		best_results.append(best)
	else:
		print("Skipping distillation (use --run_distillation to enable).")

	# 5) Direct EN adapter
	sweep_df, best = run_direct_en(args.en_adapter_dir, frozen_df, args)
	export_supplementary_table(sweep_df, os.path.join(args.output_dir, "appendix_direct_en"))
	best_results.append(best)

	# 6) Direct PL adapter baseline
	sweep_df, best = run_direct_pl(args.pl_adapter_dir, frozen_df, args)
	export_supplementary_table(sweep_df, os.path.join(args.output_dir, "appendix_direct_pl"))
	best_results.append(best)

	# Summary
	export_summary_table(best_results, args.output_dir)
	print("Done. Summary + supplementary tables exported.")


if __name__ == "__main__":
	main()
