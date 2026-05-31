#!/usr/bin/env python3
"""
Fisher-weighted DARE: non-uniform per-parameter dropout guided by diagonal Fisher
information, replacing the uniform random DARE mask from the baseline.

Core idea:
  For each EN LoRA parameter θ_i, compute the diagonal Fisher score
    F_i = (1/N) Σ_n (∂L_n/∂θ_i)²
  where L_n is the cross-entropy hate-speech loss on calibration sample n.
  Parameters with high F_i are causally important to the probe outcome and are
  kept with higher probability; low-F_i parameters are preferentially dropped.

Keep probability per element (within each parameter tensor):
  p_raw_i = fisher_low_base + (1 - fisher_low_base) * F_i / max(F)
  p_i     = clip(p_raw_i * dare_density / mean(p_raw), fisher_low_base, 1.0)

Unbiased rescale: kept element_i → w_en_i / p_i, so E[output] = w_en.

When dare_density=1.0: Fisher DARE is skipped entirely (same as baseline).
When fisher_score is all-zero (gradient did not flow to a tensor): falls back
to uniform DARE at the target density.

Why this may improve over the baseline:
  The null result of uniform per-layer DARE (Idea 2) shows that scalar-per-layer
  dropout granularity is insufficient — lower layers have near-zero EN deltas so
  any dropout is a no-op. Fisher DARE is per-parameter: it can selectively
  discard the few EN parameters that are large-but-uninformative while keeping
  the small-but-causally-important ones in upper layers.

Changes vs stella_hate_weight_merge_ties.py (baseline):
  + compute_fisher_importance(): forward-backward on EN model to collect F_i.
  + fisher_weighted_dare(): non-uniform dropout and unbiased rescale.
  + ties_dare_merge(): accepts optional fisher_score + fisher_low_base.
  + build_merged_lora_params(): passes per-tensor Fisher scores.
  + run_one_repeat(): sweeps dare_density and fisher_low_base; deduplicates
    runs where dare_density=1.0 (Fisher has no effect).
  + main(): computes Fisher once on EN model before the repeat loop.
  + New CLI: --fisher_n_samples, --fisher_seed, --fisher_low_base_grid.
  + Default dare_density_grid changed to "1.0,0.5,0.3" to exercise Fisher DARE.
  + Method name: "ties_fisher_dare".
  + output_dir default: "ties_results_fisher".
"""

import argparse
import gc
import importlib.util
import json
import math as _math
import os
import re
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import PeftModel
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL_ID = "sdadas/stella-pl"
MAX_LENGTH = 256
SEED = 42

# Per-layer probe accuracy delta (EN - PL) from stage2_per_layer.csv.
DEFAULT_PROBE_ACC_DELTA = np.array([
	-0.007542195217283165,   # layer 0
	-0.004913477596560467,   # layer 1
	-0.010944817538997231,   # layer 2
	-0.036186618342038734,   # layer 3
	-0.027509505027893177,   # layer 4
	-0.001122836939913996,   # layer 5
	 0.001878975233628455,   # layer 6
	 0.0,                    # layer 7
	 0.001891056390576851,   # layer 8
	 0.004902817752194144,   # layer 9
	 0.003772874249369229,   # layer 10
	 0.022970543296734647,   # layer 11
	 0.033908254272820850,   # layer 12
	 0.038806097430977404,   # layer 13
	 0.054641651565220270,   # layer 14
	 0.043332267348896614,   # layer 15
	 0.042593895462459510,   # layer 16
	 0.056157481434104370,   # layer 17  <- peak EN advantage
	 0.039201222328820506,   # layer 18
	 0.032421561311871350,   # layer 19
	 0.047491027964325250,   # layer 20
	 0.058796148242902135,   # layer 21  <- peak EN advantage
	 0.052771204207085140,   # layer 22
	 0.044472160039796815,   # layer 23
	 0.041075933624702410,   # layer 24
	 0.051629179547312165,   # layer 25
	 0.050116902959883470,   # layer 26
	 0.050877305191344147,   # layer 27
], dtype=np.float64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
	p = argparse.ArgumentParser(
		description="TIES/DARE merge with Fisher-weighted per-parameter dropout."
	)
	p.add_argument("--en_adapter_dir", required=True)
	p.add_argument("--pl_adapter_dir", required=True)
	p.add_argument("--frozen_test_path", default="frozen_test_strict_tuples.jsonl")
	p.add_argument("--output_dir", default="ties_results_fisher")
	p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
	p.add_argument("--batch_size", type=int, default=64)
	p.add_argument("--val_size", type=float, default=0.30)
	p.add_argument("--n_repeats", type=int, default=20)
	p.add_argument("--repeat_seed_base", type=int, default=2026)
	p.add_argument("--threshold_grid", default="0.05,0.95,0.01")
	p.add_argument("--temperature_grid", default="0.7,0.9,1.0,1.2,1.5,2.0")
	p.add_argument("--stage2_per_layer_path", default="stage2_per_layer.csv")
	# TIES/DARE sweep — pinned to winning baseline defaults except dare_density.
	p.add_argument("--global_lambda_grid", default="0.08")
	p.add_argument("--delta_scale_grid", default="0.0")
	p.add_argument("--head_lambda_grid", default="0.0")
	p.add_argument("--ties_density_grid", default="1.0")
	p.add_argument("--dare_density_grid", default="1.0,0.5,0.3",
		help="Average keep fraction for Fisher DARE (1.0=no dropout=baseline).")
	p.add_argument("--lora_b_only", action="store_true", default=False)
	# Fisher-specific args.
	p.add_argument("--fisher_n_samples", type=int, default=512,
		help="Number of calibration samples for Fisher estimation (0=disable Fisher).")
	p.add_argument("--fisher_seed", type=int, default=999,
		help="RNG seed for sampling the Fisher calibration set.")
	p.add_argument("--fisher_low_base_grid", default="0.0,0.1",
		help="Minimum keep probability for near-zero-Fisher elements. "
		     "0.0=can drop completely, 0.1=always 10%% chance of keeping.")
	# Compatibility-only (not used).
	p.add_argument("--ridge_alpha_grid", default="0.1,1.0,10.0")
	p.add_argument("--attn_lambda_grid", default="0.5,1.0")
	p.add_argument("--disagree_weight_grid", default="1.0,2.0")
	p.add_argument("--mcc_floor_delta", type=float, default=0.0)
	return p.parse_args()


def parse_float_grid(csv_str: str) -> List[float]:
	vals = [float(x.strip()) for x in csv_str.split(",") if x.strip()]
	if not vals:
		raise ValueError(f"Empty grid: {csv_str!r}")
	return vals


def parse_threshold_grid(spec: str) -> np.ndarray:
	parts = parse_float_grid(spec)
	if len(parts) != 3:
		raise ValueError("threshold_grid must be start,end,step")
	start, end, step = parts
	return np.clip(np.arange(start, end + 1e-9, step), 0.0, 1.0)


# ---------------------------------------------------------------------------
# GPU / precision
# ---------------------------------------------------------------------------

def get_precision_config() -> Dict:
	if not torch.cuda.is_available():
		return {"torch_dtype": torch.float32, "bnb_compute_dtype": torch.float32}
	major, _ = torch.cuda.get_device_capability(0)
	use_bf16 = major >= 8
	return {
		"torch_dtype": torch.bfloat16 if use_bf16 else torch.float16,
		"bnb_compute_dtype": torch.bfloat16 if use_bf16 else torch.float16,
	}


PRECISION_CONFIG = get_precision_config()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cleanup_memory():
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def clean_html_tags(text: str) -> str:
	if not isinstance(text, str):
		return ""
	text = re.sub(r"<[^>]+>", "", text)
	return re.sub(r"\s+", " ", text).strip()


def strict_map(value):
	try:
		v = int(value)
	except (TypeError, ValueError):
		return None
	return 0 if v == 0 else 1 if v in (1, 2, 3, 4) else None


def load_frozen_tuples(jsonl_path: str) -> pd.DataFrame:
	rows = []
	with open(jsonl_path, "r", encoding="utf-8") as f:
		for line in f:
			tup = json.loads(line)
			if isinstance(tup, list) and len(tup) >= 3:
				rows.append({"label": int(tup[0]), "text": str(tup[1]), "id": str(tup[2])})
	if not rows:
		raise RuntimeError(f"No valid tuples in {jsonl_path}")
	return pd.DataFrame(rows)


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
		raise RuntimeError("Too few non-frozen rows")
	return non_frozen[["id", "text", "label"]]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_bnb_config() -> BitsAndBytesConfig:
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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def sanitize_array(x: np.ndarray, clip: float = 1e6) -> np.ndarray:
	return np.clip(np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip), -clip, clip).astype(np.float32)


def softmax_np(logits: np.ndarray) -> np.ndarray:
	logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
	shifted = logits - np.max(logits, axis=-1, keepdims=True)
	e = np.exp(shifted)
	return e / e.sum(axis=-1, keepdims=True)


def run_inference(model, tokenizer, df: pd.DataFrame, batch_size: int) -> Dict:
	texts = df["text"].tolist()
	labels = df["label"].astype(int).to_numpy()
	ids = df["id"].astype(str).to_numpy()
	device = next(model.parameters()).device
	all_logits = []

	for i in range(0, len(texts), batch_size):
		batch = texts[i: i + batch_size]
		enc = tokenizer(batch, return_tensors="pt", truncation=True,
						max_length=MAX_LENGTH, padding=True)
		enc = {k: v.to(device) for k, v in enc.items()}
		with torch.inference_mode():
			logits = model(**enc).logits
		all_logits.append(logits.detach().float().cpu().numpy())

	logits_np = sanitize_array(np.concatenate(all_logits, axis=0))
	score = softmax_np(logits_np)[:, 1]
	preds = np.argmax(logits_np, axis=-1)
	return {"ids": ids, "labels": labels, "logits": logits_np, "score": score, "preds": preds}


def subset_pack(pack: Dict, idx: np.ndarray) -> Dict:
	n = pack["labels"].shape[0]
	return {
		k: v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v
		for k, v in pack.items()
	}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics_from_score(labels: np.ndarray, score: np.ndarray, threshold: float) -> Dict:
	preds = (score >= threshold).astype(int)
	ph, rh, f1h, _ = precision_recall_fscore_support(
		labels, preds, average="binary", pos_label=1, zero_division=0
	)
	return {
		"accuracy":       float(accuracy_score(labels, preds)),
		"f1_macro":       float(f1_score(labels, preds, average="macro", zero_division=0)),
		"f1_weighted":    float(f1_score(labels, preds, average="weighted", zero_division=0)),
		"precision_hate": float(ph),
		"recall_hate":    float(rh),
		"f1_hate":        float(f1h),
		"pr_auc":         float(average_precision_score(labels, score)),
		"mcc":            float(matthews_corrcoef(labels, preds)),
	}


def tune_threshold(
	score: np.ndarray,
	labels: np.ndarray,
	threshold_grid: np.ndarray,
	mcc_floor: float,
) -> Tuple[float, Dict]:
	best_thr, best_m, best_f1 = float(threshold_grid[0]), None, -1e9
	for thr in threshold_grid:
		m = metrics_from_score(labels, score, float(thr))
		if m["mcc"] + 1e-12 < mcc_floor:
			continue
		if m["f1_macro"] > best_f1:
			best_f1, best_m, best_thr = m["f1_macro"], m, float(thr)
	if best_m is None:
		for thr in threshold_grid:
			m = metrics_from_score(labels, score, float(thr))
			if m["f1_macro"] > best_f1:
				best_f1, best_m, best_thr = m["f1_macro"], m, float(thr)
	return best_thr, best_m  # type: ignore[return-value]


def fit_temperature(logits: np.ndarray, labels: np.ndarray, temp_grid: List[float]) -> float:
	best_t, best_loss = temp_grid[0], float("inf")
	eps = 1e-7
	for t in temp_grid:
		s = softmax_np(logits / t)[:, 1].astype(np.float64)
		p = np.clip(s, eps, 1.0 - eps)
		loss = -float(np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
		if loss < best_loss:
			best_loss, best_t = loss, t
	return float(best_t)


def calibrate_pack(pack: Dict, t: float) -> Dict:
	logits = sanitize_array(pack["logits"] / t)
	score = softmax_np(logits)[:, 1]
	preds = np.argmax(logits, axis=-1)
	return {**pack, "logits": logits, "score": score, "preds": preds}


# ---------------------------------------------------------------------------
# Adapter key utilities
# ---------------------------------------------------------------------------

def is_lora_key(key: str) -> bool:
	return ".lora_A." in key or ".lora_B." in key


def is_head_key(key: str) -> bool:
	return "modules_to_save" in key and (".score." in key or ".classifier." in key)


def parse_layer_from_key(key: str) -> Optional[int]:
	for pat in [r"\.layers?\.(\d+)\.", r"\.h\.(\d+)\.", r"\.blocks?\.(\d+)\."]:
		m = re.search(pat, key)
		if m:
			return int(m.group(1))
	return None


# ---------------------------------------------------------------------------
# Probe acc delta loading
# ---------------------------------------------------------------------------

def load_probe_acc_delta(path: str) -> np.ndarray:
	if path and os.path.exists(path):
		df = pd.read_csv(path)
		if "probe_acc_delta" in df.columns:
			vals = df.sort_values("layer")["probe_acc_delta"].to_numpy(dtype=np.float64)
			if len(vals) > 0:
				return vals
	return DEFAULT_PROBE_ACC_DELTA.copy()


def compute_lambda_per_layer(
	global_lambda: float,
	delta_scale: float,
	probe_acc_delta: np.ndarray,
	layer_idx: int,
) -> float:
	max_delta = float(np.max(np.abs(probe_acc_delta)))
	if max_delta < 1e-8:
		return float(np.clip(global_lambda, 0.0, 1.0))
	if layer_idx < 0 or layer_idx >= len(probe_acc_delta):
		return float(np.clip(global_lambda, 0.0, 1.0))
	delta = float(probe_acc_delta[layer_idx])
	return float(np.clip(global_lambda + delta_scale * delta / max_delta, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Fisher information computation
# ---------------------------------------------------------------------------

def compute_fisher_importance(
	model: nn.Module,
	tokenizer,
	calibration_df: pd.DataFrame,
	batch_size: int,
	n_samples: int,
	fisher_seed: int = 999,
) -> Dict[str, torch.Tensor]:
	"""
	Compute diagonal Fisher information for EN LoRA and head parameters.

	  F_i = (1/N) Σ_n (∂L_n/∂θ_i)²

	where L_n is the cross-entropy classification loss on sample n.

	Parameters with high F_i have large expected squared gradient — they are
	causally important to the hate-speech classification outcome and should be
	preserved during DARE dropout.

	Implementation notes:
	  - Only LoRA and head parameters are tracked (base 4-bit weights excluded).
	  - Gradients are temporarily enabled on target parameters; requires_grad is
	    restored to False after computation.
	  - Uses mini-batch accumulation: F_i ≈ (1/N) Σ_batch bsz * (grad_batch_i)²
	    This is an approximation of the true per-sample Fisher but cheaper.
	  - Falls back to zero Fisher (→ uniform DARE) if backward raises an error.

	Args:
	  model:          EN adapter model (4-bit NF4 quantized, eval mode).
	  tokenizer:      Tokenizer for the base model.
	  calibration_df: DataFrame with 'text' and 'label' columns (non-frozen data).
	  batch_size:     Batch size for the forward-backward passes.
	  n_samples:      Number of calibration examples to use (0 = skip Fisher).
	  fisher_seed:    RNG seed for calibration sample selection.

	Returns:
	  Dict {param_name: fisher_tensor} with same dtype (float32) and device as params.
	"""
	if n_samples <= 0:
		print("[fisher] fisher_n_samples=0 — skipping Fisher (uniform DARE will be used).")
		return {}

	rng_np = np.random.default_rng(fisher_seed)
	n = min(n_samples, len(calibration_df))
	idx = sorted(rng_np.choice(len(calibration_df), size=n, replace=False).tolist())
	sample_df = calibration_df.iloc[idx].reset_index(drop=True)

	# Identify target parameters (LoRA + head).
	target_names = {
		name for name, _ in model.named_parameters()
		if is_lora_key(name) or is_head_key(name)
	}

	# Temporarily enable gradients.
	for name, param in model.named_parameters():
		if name in target_names:
			param.requires_grad_(True)

	# Accumulators on the same device as each parameter.
	accum: Dict[str, torch.Tensor] = {}
	for name, param in model.named_parameters():
		if name in target_names:
			accum[name] = torch.zeros(param.shape, dtype=torch.float32, device=param.device)

	device = next(model.parameters()).device
	texts = sample_df["text"].tolist()
	labels_all = sample_df["label"].astype(int).tolist()
	n_seen = 0

	try:
		for i in range(0, len(texts), batch_size):
			bt = texts[i: i + batch_size]
			bl = torch.tensor(labels_all[i: i + batch_size], dtype=torch.long, device=device)
			enc = tokenizer(bt, return_tensors="pt", truncation=True,
							max_length=MAX_LENGTH, padding=True)
			enc = {k: v.to(device) for k, v in enc.items()}

			model.zero_grad()
			with torch.enable_grad():
				out = model(**enc, labels=bl)
				out.loss.backward()

			bsz = len(bt)
			for name, param in model.named_parameters():
				if name in target_names and param.grad is not None:
					# Accumulate bsz-weighted squared gradient.
					accum[name].add_(param.grad.detach().float().pow_(2).mul_(bsz))

			n_seen += bsz
			del out, bl
			cleanup_memory()

		if n_seen > 0:
			for name in accum:
				accum[name].div_(n_seen)

		print(f"[fisher] Estimated Fisher over {n_seen} samples, {len(accum)} tensors.")
		# Brief diagnostics.
		lora_b_keys = [k for k in accum if ".lora_B." in k]
		if lora_b_keys:
			for k in lora_b_keys[:3]:
				f = accum[k]
				print(f"[fisher]   {k}: max={float(f.max()):.3e}  mean={float(f.mean()):.3e}")

	except Exception as exc:
		print(f"[fisher] ERROR during Fisher computation: {exc}")
		print("[fisher] Returning zero Fisher — uniform DARE will be used as fallback.")
		for name in accum:
			accum[name].zero_()

	finally:
		model.zero_grad()
		for name, param in model.named_parameters():
			if name in target_names:
				param.requires_grad_(False)
		cleanup_memory()

	return accum


# ---------------------------------------------------------------------------
# Fisher-weighted DARE mask
# ---------------------------------------------------------------------------

def fisher_weighted_dare(
	w_en_f: torch.Tensor,
	fisher: torch.Tensor,
	dare_density: float,
	fisher_low_base: float,
	rng: np.random.Generator,
	eps: float = 1e-10,
) -> torch.Tensor:
	"""
	Apply Fisher-weighted DARE to an EN weight tensor.

	Keep probability per element:
	  p_raw_i = fisher_low_base + (1 - fisher_low_base) * F_i / max(F)
	  p_i     = clip(p_raw_i * dare_density / mean(p_raw), fisher_low_base, 1.0)

	Kept element i is rescaled by 1/p_i so that E[output_i] = w_en_i (unbiased).
	Dropped elements are set to zero.

	Falls back to uniform DARE at dare_density if F is all-zero
	(gradient did not flow through this tensor).

	Args:
	  w_en_f:          EN weight tensor (float32, any shape).
	  fisher:          Fisher score tensor (same shape as w_en_f, float32).
	  dare_density:    Target average keep fraction in (0, 1).
	  fisher_low_base: Minimum keep probability for zero-Fisher elements.
	  rng:             NumPy Generator for reproducible masks.
	  eps:             Numerical stability floor.

	Returns:
	  Modified w_en_f with Fisher-weighted dropout applied.
	"""
	f_flat = fisher.flatten().float().to(w_en_f.device)
	f_max = float(f_flat.max())

	if f_max < eps:
		# No gradient signal — fall back to uniform DARE.
		keep_np = rng.random(w_en_f.numel()) < dare_density
		mask = torch.tensor(
			keep_np.reshape(w_en_f.shape), dtype=torch.float32, device=w_en_f.device
		)
		return w_en_f * mask / (dare_density + eps)

	# Normalize Fisher to [0, 1] within this tensor.
	f_norm = f_flat / f_max

	# Raw keep probability in [fisher_low_base, 1.0].
	p_raw = fisher_low_base + (1.0 - fisher_low_base) * f_norm

	# Scale so that mean(p) ≈ dare_density.
	p_mean = float(p_raw.mean())
	if p_mean > eps:
		p_scaled = torch.clamp(p_raw * (dare_density / p_mean), fisher_low_base, 1.0)
	else:
		p_scaled = p_raw.clamp(fisher_low_base, 1.0)

	p_np = p_scaled.cpu().numpy()
	keep_np = rng.random(len(p_np)) < p_np
	keep_mask = torch.tensor(
		keep_np.reshape(w_en_f.shape), dtype=torch.float32, device=w_en_f.device
	)

	# Unbiased rescale: kept_i → w_i / p_i, dropped → 0.
	# rescale_i = keep_mask_i / p_i  (0 when dropped, 1/p_i when kept).
	rescale = keep_mask / torch.clamp(p_scaled.reshape(w_en_f.shape), eps, 1.0)
	return w_en_f * rescale


# ---------------------------------------------------------------------------
# TIES/DARE merge (Fisher-aware)
# ---------------------------------------------------------------------------

def ties_dare_merge(
	w_pl: torch.Tensor,
	w_en: torch.Tensor,
	lambda_en: float,
	ties_density: float = 1.0,
	dare_density: float = 1.0,
	rng: Optional[np.random.Generator] = None,
	fisher_score: Optional[torch.Tensor] = None,
	fisher_low_base: float = 0.1,
) -> torch.Tensor:
	"""
	TIES/DARE merge with optional Fisher-weighted dropout.

	Step 1 (DARE): When dare_density < 1.0:
	  - If fisher_score provided: Fisher-weighted dropout (high-F elements kept).
	  - Else: uniform random DARE (baseline behaviour).
	Step 2 (TIES trim):  zero elements below (1-ties_density) magnitude quantile.
	Step 3 (TIES elect): sign by lambda-weighted combination.
	Step 4 (TIES merge): keep sign-consistent elements, then LERP.
	"""
	out_dtype = w_pl.dtype
	w_pl_f = w_pl.detach().float()
	w_en_f = w_en.detach().float()

	# Step 1: DARE.
	if dare_density < 1.0:
		if fisher_score is not None and rng is not None:
			w_en_f = fisher_weighted_dare(
				w_en_f,
				fisher_score.to(w_en_f.device).float(),
				dare_density,
				fisher_low_base,
				rng,
			)
		elif rng is not None:
			# Uniform DARE fallback (fisher_score not available for this tensor).
			keep = rng.random(w_en_f.numel()) < dare_density
			drop_mask = torch.tensor(
				keep.reshape(w_en_f.shape), dtype=torch.float32, device=w_en_f.device
			)
			w_en_f = w_en_f * drop_mask / (dare_density + 1e-9)

	# Step 2: TIES trim.
	if ties_density < 1.0:
		for w_ref, name in [(w_pl_f, "pl"), (w_en_f, "en")]:
			flat = w_ref.abs().flatten()
			if flat.numel() > 1:
				q = float(torch.quantile(flat, 1.0 - ties_density))
				if name == "pl":
					w_pl_f = w_pl_f * (w_pl_f.abs() >= q).float()
				else:
					w_en_f = w_en_f * (w_en_f.abs() >= q).float()

	# Step 3: TIES elect.
	raw_merged = (1.0 - lambda_en) * w_pl_f + lambda_en * w_en_f
	elected_sign = torch.sign(raw_merged)

	# Step 4: TIES merge — LERP on sign-consistent elements.
	pl_ok = (torch.sign(w_pl_f) == elected_sign) | (w_pl_f == 0.0)
	en_ok = (torch.sign(w_en_f) == elected_sign) | (w_en_f == 0.0)
	w_pl_masked = w_pl_f * pl_ok.float()
	w_en_masked = w_en_f * en_ok.float()
	merged = (1.0 - lambda_en) * w_pl_masked + lambda_en * w_en_masked
	return merged.to(dtype=out_dtype)


def build_merged_lora_params(
	pl_lora_params: Dict[str, torch.Tensor],
	en_lora_params: Dict[str, torch.Tensor],
	probe_acc_delta: np.ndarray,
	global_lambda: float,
	delta_scale: float,
	head_lambda: float,
	ties_density: float,
	dare_density: float,
	rng: np.random.Generator,
	lora_b_only: bool = False,
	fisher_dict: Optional[Dict[str, torch.Tensor]] = None,
	fisher_low_base: float = 0.1,
) -> Dict[str, torch.Tensor]:
	"""
	Compute merged LoRA parameter dict from PL and EN adapter params.

	Fisher scores (if provided) are looked up per parameter name and passed to
	ties_dare_merge to enable non-uniform dropout. Head parameters always use
	simple LERP (no DARE — classifier head weights are well-behaved and merging
	them with Fisher DARE introduces unnecessary noise).
	"""
	merged = {}
	for name, w_pl in pl_lora_params.items():
		w_en = en_lora_params.get(name)
		if w_en is None:
			merged[name] = w_pl.clone()
			continue

		if is_head_key(name):
			# Classifier head: simple LERP.
			w_pl_f = w_pl.float()
			w_en_f = w_en.float()
			blended = (1.0 - head_lambda) * w_pl_f + head_lambda * w_en_f
			merged[name] = blended.to(w_pl.dtype)

		elif is_lora_key(name):
			if lora_b_only and ".lora_A." in name:
				merged[name] = w_pl.clone()
				continue

			layer_idx = parse_layer_from_key(name)
			lambda_l = (
				compute_lambda_per_layer(global_lambda, delta_scale, probe_acc_delta, layer_idx)
				if layer_idx is not None else global_lambda
			)

			# Look up Fisher score for this parameter (None if not computed).
			fisher_score = fisher_dict.get(name) if fisher_dict else None

			merged[name] = ties_dare_merge(
				w_pl, w_en, lambda_l, ties_density, dare_density, rng,
				fisher_score=fisher_score,
				fisher_low_base=fisher_low_base,
			)
		else:
			merged[name] = w_pl.clone()

	return merged


# ---------------------------------------------------------------------------
# In-place adapter weight swapping
# ---------------------------------------------------------------------------

def collect_adapter_params(model: nn.Module) -> Dict[str, torch.Tensor]:
	return {
		name: param.data.clone()
		for name, param in model.named_parameters()
		if is_lora_key(name) or is_head_key(name)
	}


def apply_adapter_params(model: nn.Module, params: Dict[str, torch.Tensor]) -> None:
	model_params = dict(model.named_parameters())
	for name, tensor in params.items():
		if name in model_params:
			model_params[name].data.copy_(tensor.to(
				device=model_params[name].device,
				dtype=model_params[name].dtype,
			))


# ---------------------------------------------------------------------------
# Per-repeat experiment
# ---------------------------------------------------------------------------

def run_one_repeat(
	repeat_id: int,
	split_seed: int,
	non_frozen_df: pd.DataFrame,
	pl_full: Dict,
	en_full: Dict,
	frozen_pl: Dict,
	frozen_en: Dict,
	frozen_df: pd.DataFrame,
	model: nn.Module,
	pl_lora_params: Dict[str, torch.Tensor],
	en_lora_params: Dict[str, torch.Tensor],
	threshold_grid: np.ndarray,
	temp_grid: List[float],
	global_lambda_grid: List[float],
	delta_scale_grid: List[float],
	head_lambda_grid: List[float],
	ties_density_grid: List[float],
	dare_density_grid: List[float],
	fisher_low_base_grid: List[float],
	probe_acc_delta: np.ndarray,
	mcc_floor_delta: float,
	tokenizer,
	batch_size: int,
	lora_b_only: bool = False,
	val_size: float = 0.30,
	fisher_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[List[Dict], List[Dict], Dict]:

	indices = np.arange(len(non_frozen_df))
	train_idx, tune_idx = train_test_split(
		indices, test_size=val_size, random_state=split_seed,
		stratify=non_frozen_df["label"].to_numpy(),
	)

	train_pl = subset_pack(pl_full, train_idx)
	train_en = subset_pack(en_full, train_idx)
	tune_pl  = subset_pack(pl_full, tune_idx)
	tune_en  = subset_pack(en_full, tune_idx)

	t_pl = fit_temperature(train_pl["logits"], train_pl["labels"], temp_grid)
	t_en = fit_temperature(train_en["logits"], train_en["labels"], temp_grid)

	train_pl    = calibrate_pack(train_pl, t_pl)
	tune_pl     = calibrate_pack(tune_pl, t_pl)
	frozen_pl_c = calibrate_pack(frozen_pl, t_pl)

	tune_labels   = tune_pl["labels"]
	frozen_labels = frozen_pl_c["labels"]

	pl_thr, pl_tune_m = tune_threshold(tune_pl["score"], tune_labels, threshold_grid, mcc_floor=-1.0)
	pl_tune_mcc    = float(pl_tune_m["mcc"])
	pl_frozen_m    = metrics_from_score(frozen_labels, frozen_pl_c["score"], pl_thr)
	pl_frozen_macro = float(pl_frozen_m["f1_macro"])

	tune_en_c   = calibrate_pack(tune_en, t_en)
	frozen_en_c = calibrate_pack(frozen_en, t_en)

	oracle_tune_score = np.where(
		(tune_en_c["preds"] == tune_labels) & (tune_pl["preds"] != tune_labels),
		tune_en_c["score"], tune_pl["score"],
	)
	oracle_frozen_score = np.where(
		(frozen_en_c["preds"] == frozen_labels) & (frozen_pl_c["preds"] != frozen_labels),
		frozen_en_c["score"], frozen_pl_c["score"],
	)
	oracle_thr, _ = tune_threshold(oracle_tune_score, tune_labels, threshold_grid, mcc_floor=-1.0)
	oracle_frozen_macro = float(
		metrics_from_score(frozen_labels, oracle_frozen_score, oracle_thr)["f1_macro"]
	)
	oracle_frozen_use_en_rate = float(np.mean(
		(frozen_en_c["preds"] == frozen_labels) & (frozen_pl_c["preds"] != frozen_labels)
	))

	mcc_floor = pl_tune_mcc - mcc_floor_delta

	curve_rows: List[Dict] = []
	best_row: Optional[Dict] = None
	best_soft_row: Optional[Dict] = None
	best_obj = -1e9
	best_soft_obj = -1e9
	constraints_satisfied = 0
	best_config_hparams: Optional[Dict] = None
	best_soft_config_hparams: Optional[Dict] = None

	tune_sub_df = non_frozen_df.iloc[tune_idx].reset_index(drop=True)

	for global_lambda in global_lambda_grid:
		for delta_scale in delta_scale_grid:
			for head_lambda in head_lambda_grid:
				for ties_density in ties_density_grid:
					for dare_density in dare_density_grid:
						for fisher_low_base in fisher_low_base_grid:
							# When dare_density=1.0 Fisher DARE is not applied; all
							# fisher_low_base values give identical results — run once.
							if dare_density >= 1.0 and fisher_low_base != fisher_low_base_grid[0]:
								continue

							config_tag = (
								f"gl{global_lambda:.3f}_ds{delta_scale:.2f}"
								f"_hl{head_lambda:.2f}_td{ties_density:.2f}"
								f"_dd{dare_density:.2f}_flb{fisher_low_base:.2f}"
								f"{'_lob1' if lora_b_only else ''}"
							)
							rng_cfg = np.random.default_rng(
								split_seed + abs(hash(config_tag)) % 10_000_000
							)

							merged_params = build_merged_lora_params(
								pl_lora_params, en_lora_params,
								probe_acc_delta,
								global_lambda, delta_scale, head_lambda,
								ties_density, dare_density, rng_cfg,
								lora_b_only=lora_b_only,
								fisher_dict=fisher_dict,
								fisher_low_base=fisher_low_base,
							)

							apply_adapter_params(model, merged_params)

							merged_tune_pack = run_inference(
								model, tokenizer, tune_sub_df, batch_size
							)

							apply_adapter_params(model, pl_lora_params)

							m_thr, tune_m = tune_threshold(
								merged_tune_pack["score"], tune_labels, threshold_grid, mcc_floor
							)

							row = {
								"repeat_id":      repeat_id,
								"method":         "ties_fisher_dare",
								"global_lambda":  global_lambda,
								"delta_scale":    delta_scale,
								"head_lambda":    head_lambda,
								"ties_density":   ties_density,
								"dare_density":   dare_density,
								"fisher_low_base": fisher_low_base,
								"threshold":      m_thr,
								"tune_macro_f1":       float(tune_m["f1_macro"]),
								"tune_mcc":            float(tune_m["mcc"]),
								"tune_f1_hate":        float(tune_m["f1_hate"]),
								"tune_precision_hate": float(tune_m["precision_hate"]),
								"tune_recall_hate":    float(tune_m["recall_hate"]),
								"tune_pr_auc":         float(tune_m["pr_auc"]),
								"tune_accuracy":       float(tune_m["accuracy"]),
								"frozen_macro_f1":        float("nan"),
								"frozen_f1_weighted":     float("nan"),
								"frozen_mcc":             float("nan"),
								"frozen_f1_hate":         float("nan"),
								"frozen_precision_hate":  float("nan"),
								"frozen_recall_hate":     float("nan"),
								"frozen_pr_auc":          float("nan"),
								"frozen_accuracy":        float("nan"),
								"gap_to_oracle_reduction": float("nan"),
								"pl_frozen_macro_f1":     pl_frozen_macro,
								"oracle_frozen_macro_f1": oracle_frozen_macro,
								"frozen_routed_to_en_pct":        global_lambda,
								"frozen_mean_en_attn":            global_lambda,
								"frozen_en_win_precision_routed": float("nan"),
								"frozen_disagreement_acc":        float("nan"),
							}
							curve_rows.append(row)

							hparams = dict(
								global_lambda=global_lambda, delta_scale=delta_scale,
								head_lambda=head_lambda, ties_density=ties_density,
								dare_density=dare_density, fisher_low_base=fisher_low_base,
								config_tag=config_tag, threshold=m_thr,
							)

							if float(tune_m["mcc"]) + 1e-12 >= mcc_floor:
								constraints_satisfied += 1
								if float(tune_m["f1_macro"]) > best_obj:
									best_obj = float(tune_m["f1_macro"])
									best_row = row
									best_config_hparams = hparams

							v_mcc = max(0.0, mcc_floor - float(tune_m["mcc"]))
							soft_obj = float(tune_m["f1_macro"]) - 5.0 * v_mcc
							if soft_obj > best_soft_obj:
								best_soft_obj = soft_obj
								best_soft_row = row
								best_soft_config_hparams = hparams

	used_soft_fallback = False
	if best_row is None:
		best_row = best_soft_row if best_soft_row is not None else (
			max(curve_rows, key=lambda r: r["tune_macro_f1"]) if curve_rows else None
		)
		best_config_hparams = best_soft_config_hparams
		used_soft_fallback = True

	if best_row is not None and best_config_hparams is not None:
		rng_best = np.random.default_rng(
			split_seed + abs(hash(best_config_hparams["config_tag"])) % 10_000_000
		)
		best_merged = build_merged_lora_params(
			pl_lora_params, en_lora_params, probe_acc_delta,
			best_config_hparams["global_lambda"],
			best_config_hparams["delta_scale"],
			best_config_hparams["head_lambda"],
			best_config_hparams["ties_density"],
			best_config_hparams["dare_density"],
			rng_best,
			lora_b_only=lora_b_only,
			fisher_dict=fisher_dict,
			fisher_low_base=best_config_hparams["fisher_low_base"],
		)
		apply_adapter_params(model, best_merged)
		best_frozen_pack = run_inference(model, tokenizer, frozen_df, batch_size)
		apply_adapter_params(model, pl_lora_params)

		best_thr = best_config_hparams["threshold"]
		frozen_m = metrics_from_score(frozen_labels, best_frozen_pack["score"], best_thr)
		gap_red = (float(frozen_m["f1_macro"]) - pl_frozen_macro) / max(
			1e-8, oracle_frozen_macro - pl_frozen_macro
		)
		best_row.update({
			"frozen_macro_f1":        float(frozen_m["f1_macro"]),
			"frozen_f1_weighted":     float(frozen_m["f1_weighted"]),
			"frozen_mcc":             float(frozen_m["mcc"]),
			"frozen_f1_hate":         float(frozen_m["f1_hate"]),
			"frozen_precision_hate":  float(frozen_m["precision_hate"]),
			"frozen_recall_hate":     float(frozen_m["recall_hate"]),
			"frozen_pr_auc":          float(frozen_m["pr_auc"]),
			"frozen_accuracy":        float(frozen_m["accuracy"]),
			"gap_to_oracle_reduction": float(gap_red),
			"frozen_routed_to_en_pct": best_config_hparams["global_lambda"],
			"frozen_mean_en_attn":     best_config_hparams["global_lambda"],
		})

	repeat_rows = [best_row] if best_row is not None else []

	diagnostics = {
		"repeat_id":               repeat_id,
		"split_seed":              split_seed,
		"best_dare_density":       best_config_hparams["dare_density"] if best_config_hparams else float("nan"),
		"best_fisher_low_base":    best_config_hparams["fisher_low_base"] if best_config_hparams else float("nan"),
		"pl_tune_threshold":       float(pl_thr),
		"pl_tune_mcc":             float(pl_tune_mcc),
		"pl_frozen_macro_f1":      pl_frozen_macro,
		"oracle_frozen_macro_f1":  oracle_frozen_macro,
		"oracle_frozen_use_en_rate": oracle_frozen_use_en_rate,
		"temperature_pl":          float(t_pl),
		"temperature_en":          float(t_en),
		"mcc_floor":               mcc_floor,
		"constraints_satisfied":   constraints_satisfied,
		"used_soft_fallback":      int(used_soft_fallback),
	}

	return repeat_rows, curve_rows, diagnostics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_robustness(per_repeat_best: pd.DataFrame) -> pd.DataFrame:
	rows = []
	for method, g in per_repeat_best.groupby("method"):
		rows.append({
			"method":               method,
			"n_repeats":            int(len(g)),
			"macro_f1_mean":        float(g["frozen_macro_f1"].mean()),
			"macro_f1_std":         float(g["frozen_macro_f1"].std(ddof=0)),
			"f1_weighted_mean":     float(g["frozen_f1_weighted"].mean()) if "frozen_f1_weighted" in g.columns else float("nan"),
			"f1_weighted_std":      float(g["frozen_f1_weighted"].std(ddof=0)) if "frozen_f1_weighted" in g.columns else float("nan"),
			"mcc_mean":             float(g["frozen_mcc"].mean()),
			"mcc_std":              float(g["frozen_mcc"].std(ddof=0)),
			"f1_hate_mean":         float(g["frozen_f1_hate"].mean()),
			"f1_hate_std":          float(g["frozen_f1_hate"].std(ddof=0)),
			"precision_hate_mean":  float(g["frozen_precision_hate"].mean()),
			"precision_hate_std":   float(g["frozen_precision_hate"].std(ddof=0)),
			"recall_hate_mean":     float(g["frozen_recall_hate"].mean()),
			"recall_hate_std":      float(g["frozen_recall_hate"].std(ddof=0)),
			"pr_auc_mean":          float(g["frozen_pr_auc"].mean()),
			"pr_auc_std":           float(g["frozen_pr_auc"].std(ddof=0)),
			"accuracy_mean":        float(g["frozen_accuracy"].mean()),
			"accuracy_std":         float(g["frozen_accuracy"].std(ddof=0)),
			"routed_to_en_mean":    float(g["frozen_routed_to_en_pct"].mean()),
			"mean_en_attn_mean":    float(g["frozen_mean_en_attn"].mean()),
			"en_win_precision_mean": float("nan"),
			"disagreement_acc_mean": float("nan"),
			"gap_reduction_mean":    float(g["gap_to_oracle_reduction"].mean()),
			"pl_frozen_macro_f1_mean":     float(g["pl_frozen_macro_f1"].mean()),
			"oracle_frozen_macro_f1_mean": float(g["oracle_frozen_macro_f1"].mean()),
		})
	return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------

METRIC_ORDER_LATEX = [
	("frozen_macro_f1",   "Macro F1"),
	("frozen_mcc",        "MCC"),
	("frozen_f1_hate",    "F1 (hate)"),
	("frozen_pr_auc",     "PR-AUC"),
	("frozen_accuracy",   "Accuracy"),
	("gap_to_oracle_reduction", "Oracle gap red."),
]

ROBUST_METRIC_ORDER = [
	("macro_f1_mean",  "macro_f1_std",  "Macro F1"),
	("mcc_mean",       "mcc_std",       "MCC"),
	("f1_hate_mean",   "f1_hate_std",   "F1 (hate)"),
	("pr_auc_mean",    "pr_auc_std",    "PR-AUC"),
	("accuracy_mean",  "accuracy_std",  "Accuracy"),
	("gap_reduction_mean", None,        "Oracle gap red."),
	("pl_frozen_macro_f1_mean", None,   "PL baseline"),
	("oracle_frozen_macro_f1_mean", None, "Oracle ceiling"),
]

FINAL_EVAL_METRICS = [
	("accuracy_mean",       "accuracy_std",       "Accuracy"),
	("macro_f1_mean",       "macro_f1_std",       "Macro F1"),
	("f1_weighted_mean",    "f1_weighted_std",    "Weighted F1"),
	("precision_hate_mean", "precision_hate_std", "Precision (hate)"),
	("recall_hate_mean",    "recall_hate_std",    "Recall (hate)"),
	("f1_hate_mean",        "f1_hate_std",        "F1 (hate)"),
	("pr_auc_mean",         "pr_auc_std",         "PR AUC"),
	("mcc_mean",            "mcc_std",            "MCC"),
]


def _fmt(v) -> str:
	if v is None or (isinstance(v, float) and np.isnan(v)):
		return "--"
	return f"{v:.4f}"


def _t95(n: int) -> float:
	if n <= 1:
		return float("nan")
	df = n - 1
	_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
			  6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
			  11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
			  16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
			  25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
	if df in _table:
		return _table[df]
	keys = sorted(_table)
	for i in range(len(keys) - 1):
		if keys[i] < df < keys[i + 1]:
			t0, t1 = _table[keys[i]], _table[keys[i + 1]]
			a = (df - keys[i]) / (keys[i + 1] - keys[i])
			return t0 + a * (t1 - t0)
	return 1.960


def export_latex_sweep(curve_df: pd.DataFrame, path: str) -> None:
	sweep_cols = (
		["repeat_id", "global_lambda", "delta_scale", "head_lambda",
		 "ties_density", "dare_density", "fisher_low_base"]
		+ [k for k, _ in METRIC_ORDER_LATEX]
	)
	cols = [c for c in sweep_cols if c in curve_df.columns]
	tex_rows = [
		" & ".join(
			_fmt(row[c]) if isinstance(row[c], float) else str(row[c]) for c in cols
		) + r" \\"
		for _, row in curve_df[cols].iterrows()
	]
	header = " & ".join(c for c in cols) + r" \\"
	lines = (
		[r"\begin{longtable}{" + "l" * len(cols) + "}", r"\toprule", header,
		 r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
		+ tex_rows
		+ [r"\end{longtable}"]
	)
	with open(path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")


def export_latex_final_eval(robust_df: pd.DataFrame, path: str) -> None:
	col_headers = [m[2] for m in FINAL_EVAL_METRICS]
	header_str = "Method & " + " & ".join(col_headers) + r" \\"
	lines = [
		r"\begin{table}[ht]", r"\centering",
		r"\caption{Final evaluation on the frozen held-out test set (" +
		r"mean with 95\% CI across repeated train/tune splits).}",
		r"\label{tab:final_evaluation_fisher}",
		r"\begin{tabular}{l" + "r" * len(FINAL_EVAL_METRICS) + "}",
		r"\toprule", header_str, r"\midrule",
	]
	for _, row in robust_df.iterrows():
		n = int(row.get("n_repeats", 20))
		t = _t95(n)
		cells = [str(row["method"])]
		for mean_col, std_col, _ in FINAL_EVAL_METRICS:
			mv = row.get(mean_col, float("nan"))
			sv = row.get(std_col, float("nan"))
			if any(isinstance(v, float) and _math.isnan(v) for v in [mv, sv]):
				cells.append("--")
			else:
				hw = t * float(sv) / _math.sqrt(n)
				lo = float(mv) - hw
				hi = float(mv) + hw
				cells.append(f"{float(mv):.4f} ({lo:.4f}--{hi:.4f})")
		lines.append(" & ".join(cells) + r" \\")
	lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
	with open(path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")


def export_latex_robustness(robust_df: pd.DataFrame, path: str) -> None:
	lines = [
		r"\begin{table}[ht]", r"\centering",
		r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
		"Method & Macro F1 & MCC & F1 (hate) & PR-AUC & Accuracy "
		"& Oracle gap & PL baseline & Oracle ceiling \\\\",
		r"\midrule",
	]
	for _, row in robust_df.iterrows():
		def msd(m, s):
			if s is None:
				return _fmt(row.get(m, float("nan")))
			sv = row.get(s, float("nan"))
			return f"{_fmt(row.get(m, float('nan')))} {{\\small ±{_fmt(sv)}}}"

		cells = [str(row["method"])] + [
			msd(m, s) for m, s, _ in ROBUST_METRIC_ORDER
		]
		lines.append(" & ".join(cells) + r" \\")
	lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
	with open(path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
	args = parse_args()
	os.makedirs(args.output_dir, exist_ok=True)

	threshold_grid       = parse_threshold_grid(args.threshold_grid)
	temp_grid            = [float(x) for x in args.temperature_grid.split(",")]
	global_lambda_grid   = parse_float_grid(args.global_lambda_grid)
	delta_scale_grid     = parse_float_grid(args.delta_scale_grid)
	head_lambda_grid     = parse_float_grid(args.head_lambda_grid)
	ties_density_grid    = parse_float_grid(args.ties_density_grid)
	dare_density_grid    = parse_float_grid(args.dare_density_grid)
	fisher_low_base_grid = parse_float_grid(args.fisher_low_base_grid)

	probe_acc_delta = load_probe_acc_delta(args.stage2_per_layer_path)
	print(f"[info] probe_acc_delta: {len(probe_acc_delta)} layers, "
		  f"max={np.max(probe_acc_delta):.4f} at layer {np.argmax(probe_acc_delta)}")

	# Load data.
	frozen_df = load_frozen_tuples(args.frozen_test_path)
	frozen_ids = set(frozen_df["id"].astype(str))
	non_frozen_df = load_non_frozen_polish_data(frozen_ids, hf_token=args.hf_token)
	print(f"[info] non_frozen={len(non_frozen_df)}, frozen={len(frozen_df)}")

	# ------------------------------------------------------------------ #
	# Step 1: Load EN model, run inference, compute Fisher, extract params. #
	# ------------------------------------------------------------------ #
	print("[info] Loading EN adapter model...")
	en_model, tokenizer = load_model(args.en_adapter_dir, hf_token=args.hf_token)

	print("[info] Running EN inference on full dataset...")
	en_full   = run_inference(en_model, tokenizer, non_frozen_df, args.batch_size)
	frozen_en = run_inference(en_model, tokenizer, frozen_df,     args.batch_size)

	# Fisher computation: done once on EN model before it is deleted.
	print(f"[info] Computing Fisher importance "
		  f"(n_samples={args.fisher_n_samples}, seed={args.fisher_seed})...")
	fisher_dict = compute_fisher_importance(
		en_model, tokenizer, non_frozen_df,
		batch_size=args.batch_size,
		n_samples=args.fisher_n_samples,
		fisher_seed=args.fisher_seed,
	)

	en_lora_params = collect_adapter_params(en_model)
	print(f"[info] EN adapter params: {len(en_lora_params)} tensors")

	# Fisher coverage check.
	fisher_lora_keys = {k for k in fisher_dict if is_lora_key(k)}
	print(f"[info] Fisher coverage: {len(fisher_lora_keys)}/{sum(is_lora_key(k) for k in en_lora_params)} LoRA tensors")

	del en_model
	cleanup_memory()

	# ------------------------------------------------------------------ #
	# Step 2: Load PL model (stays loaded throughout all repeats).       #
	# ------------------------------------------------------------------ #
	print("[info] Loading PL adapter model...")
	pl_model, _ = load_model(args.pl_adapter_dir, hf_token=args.hf_token)

	print("[info] Running PL inference on full dataset...")
	pl_full   = run_inference(pl_model, tokenizer, non_frozen_df, args.batch_size)
	frozen_pl = run_inference(pl_model, tokenizer, frozen_df,     args.batch_size)

	pl_lora_params = collect_adapter_params(pl_model)
	print(f"[info] PL adapter params: {len(pl_lora_params)} tensors")

	missing_in_en = set(pl_lora_params) - set(en_lora_params)
	if missing_in_en:
		print(f"[warn] {len(missing_in_en)} PL keys not in EN: {list(missing_in_en)[:3]} ...")

	# Effective config count (dare_density=1.0 deduplicated across fisher_low_base).
	n_flb = len(fisher_low_base_grid)
	n_dd  = len(dare_density_grid)
	n_dd1 = sum(1 for d in dare_density_grid if d >= 1.0)   # collapsing fisher_low_base
	n_configs = (len(global_lambda_grid) * len(delta_scale_grid)
				 * len(head_lambda_grid) * len(ties_density_grid)
				 * (n_dd1 + (n_dd - n_dd1) * n_flb))
	print(f"[info] Sweep: ~{n_configs} configs × {args.n_repeats} repeats "
		  f"[Fisher DARE, dare_density ∈ {dare_density_grid}, "
		  f"fisher_low_base ∈ {fisher_low_base_grid}]")

	# ------------------------------------------------------------------ #
	# Step 3: Repeat loop.                                               #
	# ------------------------------------------------------------------ #
	all_repeat_rows: List[Dict] = []
	all_curve_rows:  List[Dict] = []
	all_diagnostics: List[Dict] = []

	for rep in range(args.n_repeats):
		split_seed = args.repeat_seed_base + rep
		print(f"\n[repeat {rep}] seed={split_seed}")

		repeat_rows, curve_rows, diagnostics = run_one_repeat(
			repeat_id=rep,
			split_seed=split_seed,
			non_frozen_df=non_frozen_df,
			pl_full=pl_full,
			en_full=en_full,
			frozen_pl=frozen_pl,
			frozen_en=frozen_en,
			frozen_df=frozen_df,
			model=pl_model,
			pl_lora_params=pl_lora_params,
			en_lora_params=en_lora_params,
			threshold_grid=threshold_grid,
			temp_grid=temp_grid,
			global_lambda_grid=global_lambda_grid,
			delta_scale_grid=delta_scale_grid,
			head_lambda_grid=head_lambda_grid,
			ties_density_grid=ties_density_grid,
			dare_density_grid=dare_density_grid,
			fisher_low_base_grid=fisher_low_base_grid,
			probe_acc_delta=probe_acc_delta,
			mcc_floor_delta=args.mcc_floor_delta,
			tokenizer=tokenizer,
			batch_size=args.batch_size,
			lora_b_only=args.lora_b_only,
			val_size=args.val_size,
			fisher_dict=fisher_dict,
		)

		all_repeat_rows.extend(repeat_rows)
		all_curve_rows.extend(curve_rows)
		all_diagnostics.append(diagnostics)

		if repeat_rows:
			best = repeat_rows[0]
			print(
				f"[repeat {rep}] best: "
				f"gl={best['global_lambda']:.2f} td={best['ties_density']:.2f} "
				f"dd={best['dare_density']:.2f} flb={best['fisher_low_base']:.2f}  "
				f"frozen_f1={best['frozen_macro_f1']:.4f} "
				f"(PL={best['pl_frozen_macro_f1']:.4f})"
			)
		print(
			f"[repeat {rep}] PL={diagnostics['pl_frozen_macro_f1']:.4f} "
			f"oracle={diagnostics['oracle_frozen_macro_f1']:.4f} "
			f"constraints_ok={diagnostics['constraints_satisfied']} "
			f"best_dd={diagnostics['best_dare_density']} "
			f"best_flb={diagnostics['best_fisher_low_base']}"
		)

	# ------------------------------------------------------------------ #
	# Step 4: Save results.                                              #
	# ------------------------------------------------------------------ #
	per_repeat_df = pd.DataFrame(all_repeat_rows)
	curve_df      = pd.DataFrame(all_curve_rows)
	diag_df       = pd.DataFrame(all_diagnostics)

	wins = 0
	method_name = "ties_fisher_dare"
	if "frozen_macro_f1" in per_repeat_df.columns and "pl_frozen_macro_f1" in per_repeat_df.columns:
		wins = int((per_repeat_df["frozen_macro_f1"] > per_repeat_df["pl_frozen_macro_f1"]).sum())
	wins_df = pd.DataFrame([{"method": method_name, "n_repeats_beat_pl": wins}])

	robust_df = aggregate_robustness(per_repeat_df) if len(per_repeat_df) > 0 else pd.DataFrame()

	per_repeat_df.to_csv(os.path.join(args.output_dir, "per_repeat_best.csv"), index=False)
	curve_df.to_csv(os.path.join(args.output_dir, "coverage_utility_curve.csv"), index=False)
	diag_df.to_csv(os.path.join(args.output_dir, "repeat_diagnostics.csv"), index=False)
	wins_df.to_csv(os.path.join(args.output_dir, "wins_over_pl.csv"), index=False)
	if len(robust_df) > 0:
		robust_df.to_csv(os.path.join(args.output_dir, "robustness_summary.csv"), index=False)

	if len(curve_df) > 0:
		export_latex_sweep(curve_df, os.path.join(args.output_dir, "appendix_sweep.tex"))
	if len(robust_df) > 0:
		export_latex_robustness(robust_df, os.path.join(args.output_dir, "table_robustness.tex"))
		export_latex_final_eval(robust_df, os.path.join(args.output_dir, "table_final_evaluation.tex"))

	print("\n===== Summary =====")
	print(wins_df.to_string(index=False))
	if len(robust_df) > 0:
		print(robust_df[["method", "macro_f1_mean", "macro_f1_std",
						  "pl_frozen_macro_f1_mean", "oracle_frozen_macro_f1_mean",
						  "gap_reduction_mean"]].to_string(index=False))

	# Fisher-specific summary: which dare_density/fisher_low_base was selected most.
	if len(diag_df) > 0 and "best_dare_density" in diag_df.columns:
		sel = diag_df.groupby(["best_dare_density", "best_fisher_low_base"]).size()
		print("\nSelected (dare_density, fisher_low_base) frequency:")
		print(sel.to_string())

	print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
	main()
