#!/usr/bin/env python3
"""
TIES/DARE weight-space merging of PL and EN LoRA adapters for Polish hate-speech detection.

Why this approach is more promising than prior methods:
=======================================================
1. No per-sample routing decision.
   All previous failures (attention fusion v1/v2, gating, ablation routing) share a root
   cause: routing oracle-win cases (~5-6% of data) at inference time is fundamentally
   unsolvable from pooled embeddings. Probe AUC~0.60 and frozen precision~11% confirm
   the signal is not separable at the embedding level.  TIES/DARE bypasses routing
   entirely: a single merged model makes predictions, so there is no precision-collapse
   from wrong routing.

2. Data-driven layer weighting via probe_acc_delta.
   stage2_per_layer.csv shows a clear, monotonically increasing EN advantage starting at
   layer 6 and peaking at layers 17/21 (~+5.6pp accuracy delta).  The merged lambda is
   set per-layer using this empirical signal, unlike the prior task_vector_scaled_state
   which used stage1_cosines with a BACKWARDS penalty formula that reduced EN contribution
   exactly where cosines were high (i.e., where the hate directions were best aligned).

3. TIES sign election removes parameter interference.
   Task-vector addition (w_pl + alpha * w_en) accumulates conflicting gradient directions
   and grows weight magnitudes unboundedly.  TIES (Trim-Elect-Merge) resolves sign
   conflicts: elements where PL and EN disagree in sign are zeroed out before merging,
   so the merged adapter only retains coherent structure.

4. DARE dropout reduces over-contribution from EN.
   Randomly dropping (1-p) fraction of EN delta weights and rescaling surviving weights
   by 1/p provides an unbiased estimator of the EN contribution while reducing the risk
   of EN features overpowering Polish-specific patterns learned by PL.

5. In-place weight swapping — no model reloads.
   The base model (4-bit NF4 quantized) is loaded once.  Each config swaps LoRA parameter
   tensors in-place (memory-mapped ops), then restores them.  This is ~100× faster than
   the prior pattern of writing temp dirs and calling PeftModel.from_pretrained() per run.

Prior implementation mistakes fixed here:
  ✗  stage1_cosines penalty was backwards (high cosine → low EN weight, should be high).
  ✗  Layer thresholds (11-14 mid, 15-22 upper) were arbitrary.
  ✗  No TIES sign election → conflicting signs caused mutual cancellation.
  ✗  Unbounded task-vector addition (w_pl + alpha*w_en, no rescaling).
  ✗  Full model reload per sweep config.

Algorithm (per LoRA tensor key):
  1. DARE dropout: mask (1 - dare_density) fraction of EN delta, rescale survivors by 1/p.
  2. TIES trim:    zero elements below the (1 - ties_density) magnitude quantile.
  3. TIES elect:   compute sign-weighted sum → elected sign per element.
  4. TIES merge:   zero elements conflicting with elected sign, then interpolate.
  lambda_l = clip(global_lambda + delta_scale * probe_acc_delta_l / max_delta, 0, 1)
  head weights use a separate head_lambda.

Output files match the schema of stella_hate_adapter_fusion_v2.py for direct comparison.
"""

import argparse
import gc
import importlib.util
import json
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
# Positive = EN better at that layer.  Used for layer-adaptive lambda computation.
# Loaded from file at runtime if --stage2_per_layer_path is provided; this is the fallback.
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
	 0.056157481434104370,   # layer 17  ← peak EN advantage
	 0.039201222328820506,   # layer 18
	 0.032421561311871350,   # layer 19
	 0.047491027964325250,   # layer 20
	 0.058796148242902135,   # layer 21  ← peak EN advantage
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
		description="TIES/DARE weight-space merging of PL and EN LoRA adapters."
	)
	# Core adapter paths (same as all prior scripts).
	p.add_argument("--en_adapter_dir", required=True)
	p.add_argument("--pl_adapter_dir", required=True)
	p.add_argument("--frozen_test_path", default="frozen_test_strict_tuples.jsonl")
	p.add_argument("--output_dir", default="ties_results")
	p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
	p.add_argument("--batch_size", type=int, default=64)
	p.add_argument("--val_size", type=float, default=0.15)
	p.add_argument("--n_repeats", type=int, default=5)
	p.add_argument("--repeat_seed_base", type=int, default=2026)
	p.add_argument("--threshold_grid", default="0.05,0.95,0.01")
	p.add_argument("--temperature_grid", default="0.7,0.9,1.0,1.2,1.5,2.0")
	# Layer-adaptive lambda.
	p.add_argument("--stage2_per_layer_path", default="stage2_per_layer.csv",
		help="Path to stage2_per_layer.csv for probe_acc_delta. Falls back to hardcoded values.")
	# TIES/DARE sweep parameters.
	p.add_argument("--global_lambda_grid", default="0.05,0.1,0.2,0.3,0.5",
		help="Overall EN blend coefficient (0=pure PL, 1=pure EN).")
	p.add_argument("--delta_scale_grid", default="0.0,0.5",
		help="Amplification of layer-adaptive variation. 0=uniform lambda across layers.")
	p.add_argument("--head_lambda_grid", default="0.0,0.2",
		help="Separate blend coefficient for classifier head weights.")
	p.add_argument("--ties_density_grid", default="1.0,0.7",
		help="Fraction of weights to keep after TIES trimming (1.0=no trim).")
	p.add_argument("--dare_density_grid", default="1.0",
		help="Fraction of EN delta weights kept by DARE dropout (1.0=no dropout).")
	p.add_argument("--lora_b_only", action="store_true", default=False,
		help="Merge only lora_B matrices; keep PL lora_A unchanged. "
		     "More principled: lora_A is the shared input projection, "
		     "lora_B is the task-specific output projection.")
	# Compatibility-only (not used, preserved for CLI parity with prior scripts).
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
	"""Load 4-bit quantized base model with a LoRA adapter.  Returns (model, tokenizer)."""
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
	"""Run batched inference; return dict with logits, score, preds, labels, ids."""
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
	"""Extract transformer layer index from a model parameter name."""
	for pat in [r"\.layers?\.(\d+)\.", r"\.h\.(\d+)\.", r"\.blocks?\.(\d+)\."]:
		m = re.search(pat, key)
		if m:
			return int(m.group(1))
	return None


# ---------------------------------------------------------------------------
# Probe acc delta loading
# ---------------------------------------------------------------------------

def load_probe_acc_delta(path: str) -> np.ndarray:
	"""Load per-layer probe_acc_delta from stage2 CSV.  Falls back to hardcoded defaults."""
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
	"""
	Layer-adaptive EN blend coefficient.

	lambda_l = clip(global_lambda + delta_scale * delta_l / max_delta_magnitude, 0, 1)

	When delta_scale=0: uniform lambda = global_lambda for all layers.
	When delta_scale>0: layers with higher EN probe advantage get higher EN blend.
	"""
	max_delta = float(np.max(np.abs(probe_acc_delta)))
	if max_delta < 1e-8:
		return float(np.clip(global_lambda, 0.0, 1.0))
	if layer_idx < 0 or layer_idx >= len(probe_acc_delta):
		return float(np.clip(global_lambda, 0.0, 1.0))
	delta = float(probe_acc_delta[layer_idx])
	return float(np.clip(global_lambda + delta_scale * delta / max_delta, 0.0, 1.0))


# ---------------------------------------------------------------------------
# TIES/DARE merge
# ---------------------------------------------------------------------------

def ties_dare_merge(
	w_pl: torch.Tensor,
	w_en: torch.Tensor,
	lambda_en: float,
	ties_density: float = 1.0,
	dare_density: float = 1.0,
	rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
	"""
	Merge two LoRA weight tensors using TIES sign election + optional DARE dropout.

	DARE dropout (Yu et al., 2023):
	  Randomly zero (1 - dare_density) fraction of EN delta elements, rescale
	  survivors by 1/dare_density to keep the expectation unbiased.

	TIES trim-elect-merge (Yadav et al., 2023):
	  Trim:  zero out elements below (1 - ties_density) magnitude quantile.
	  Elect: compute majority sign from lambda-weighted sum.
	  Merge: keep only elements agreeing with elected sign, then interpolate.

	Args:
	  w_pl, w_en:   weight tensors from PL and EN adapters (same shape).
	  lambda_en:    EN interpolation weight (0 = pure PL, 1 = pure EN).
	  ties_density: fraction to keep after trim (1.0 = no trim).
	  dare_density: fraction of EN weights to keep (1.0 = no dropout).
	  rng:          numpy rng for reproducible DARE dropout.
	"""
	# Work in float32 for numerical stability; restore dtype at end.
	out_dtype = w_pl.dtype
	w_pl_f = w_pl.detach().float()
	w_en_f = w_en.detach().float()

	# Step 1: DARE on EN delta (task vector = w_en since adapters start near 0 for lora_B).
	if dare_density < 1.0:
		if rng is None:
			drop_mask = torch.ones_like(w_en_f)
		else:
			keep = rng.random(w_en_f.numel()) < dare_density
			# Create mask on same device as weight tensor to avoid cross-device ops.
			drop_mask = torch.tensor(
				keep.reshape(w_en_f.shape), dtype=torch.float32, device=w_en_f.device
			)
		w_en_f = w_en_f * drop_mask / (dare_density + 1e-9)

	# Step 2: TIES trim — zero out low-magnitude elements.
	if ties_density < 1.0:
		for w_ref, name in [(w_pl_f, "pl"), (w_en_f, "en")]:
			flat = w_ref.abs().flatten()
			if flat.numel() > 1:
				# Keep top-ties_density fraction by magnitude.
				q = float(torch.quantile(flat, 1.0 - ties_density))
				if name == "pl":
					w_pl_f = w_pl_f * (w_pl_f.abs() >= q).float()
				else:
					w_en_f = w_en_f * (w_en_f.abs() >= q).float()

	# Step 3: TIES elect — sign by lambda-weighted combination.
	raw_merged = (1.0 - lambda_en) * w_pl_f + lambda_en * w_en_f
	elected_sign = torch.sign(raw_merged)  # 0 where raw_merged==0 → those contribute 0

	# Step 4: TIES merge — keep only sign-consistent elements, then interpolate.
	# A weight element "agrees" if sign(element) == elected_sign or element == 0.
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
) -> Dict[str, torch.Tensor]:
	"""
	Compute merged LoRA parameter dict from PL and EN adapter params.

	Applies:
	  - Layer-adaptive lambda for LoRA body weights (derived from probe_acc_delta).
	  - Separate head_lambda for classifier head (modules_to_save).
	  - TIES/DARE merge for all LoRA tensors.
	  - Simple linear interpolation for head weights (head is well-behaved).

	When lora_b_only=True:
	  - lora_A matrices are NOT merged; PL lora_A is always used unchanged.
	  - Only lora_B matrices receive TIES/DARE merging.
	  Rationale: lora_A is the shared input projection (rank adapter); lora_B carries
	  the task-specific output direction. Merging only lora_B avoids corrupting the
	  projection geometry while still blending task knowledge.

	Keys not present in both adapters pass through as PL values.
	"""
	merged = {}
	for name, w_pl in pl_lora_params.items():
		w_en = en_lora_params.get(name)
		if w_en is None:
			merged[name] = w_pl.clone()
			continue

		if is_head_key(name):
			# Classifier head: simple linear interpolation — no sign conflicts expected.
			w_pl_f = w_pl.float()
			w_en_f = w_en.float()
			blended = (1.0 - head_lambda) * w_pl_f + head_lambda * w_en_f
			merged[name] = blended.to(w_pl.dtype)

		elif is_lora_key(name):
			# When lora_b_only: skip lora_A entirely (keep PL).
			if lora_b_only and ".lora_A." in name:
				merged[name] = w_pl.clone()
				continue

			# LoRA body: TIES/DARE with layer-adaptive lambda.
			layer_idx = parse_layer_from_key(name)
			if layer_idx is not None:
				lambda_l = compute_lambda_per_layer(global_lambda, delta_scale,
													probe_acc_delta, layer_idx)
			else:
				lambda_l = global_lambda

			merged[name] = ties_dare_merge(
				w_pl, w_en, lambda_l, ties_density, dare_density, rng
			)
		else:
			# Non-LoRA, non-head parameters: pass through PL.
			merged[name] = w_pl.clone()

	return merged


# ---------------------------------------------------------------------------
# In-place adapter weight swapping
# ---------------------------------------------------------------------------

def collect_adapter_params(model: nn.Module) -> Dict[str, torch.Tensor]:
	"""Return {param_name: tensor_clone} for all LoRA and head parameters."""
	return {
		name: param.data.clone()
		for name, param in model.named_parameters()
		if is_lora_key(name) or is_head_key(name)
	}


def apply_adapter_params(model: nn.Module, params: Dict[str, torch.Tensor]) -> None:
	"""Update model LoRA/head parameters in-place from a dict."""
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
	# Stored model state for weight swapping
	model: nn.Module,
	pl_lora_params: Dict[str, torch.Tensor],
	en_lora_params: Dict[str, torch.Tensor],
	# Hyperparameter grids
	threshold_grid: np.ndarray,
	temp_grid: List[float],
	global_lambda_grid: List[float],
	delta_scale_grid: List[float],
	head_lambda_grid: List[float],
	ties_density_grid: List[float],
	dare_density_grid: List[float],
	probe_acc_delta: np.ndarray,
	mcc_floor_delta: float,
	tokenizer,
	batch_size: int,
	lora_b_only: bool = False,
	val_size: float = 0.15,
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

	# Temperature calibration on train split.
	t_pl = fit_temperature(train_pl["logits"], train_pl["labels"], temp_grid)
	t_en = fit_temperature(train_en["logits"], train_en["labels"], temp_grid)

	train_pl  = calibrate_pack(train_pl, t_pl)
	tune_pl   = calibrate_pack(tune_pl, t_pl)
	frozen_pl_c = calibrate_pack(frozen_pl, t_pl)

	tune_labels    = tune_pl["labels"]
	frozen_labels  = frozen_pl_c["labels"]

	# PL baseline: calibrated PL on frozen.
	pl_thr, pl_tune_m = tune_threshold(tune_pl["score"], tune_labels, threshold_grid, mcc_floor=-1.0)
	pl_tune_mcc = float(pl_tune_m["mcc"])
	pl_frozen_m = metrics_from_score(frozen_labels, frozen_pl_c["score"], pl_thr)
	pl_frozen_macro = float(pl_frozen_m["f1_macro"])

	# Oracle upper bound (EN correct & PL wrong → use EN; otherwise use PL).
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

	# Sweep.
	curve_rows: List[Dict] = []
	best_row: Optional[Dict] = None
	best_soft_row: Optional[Dict] = None
	best_obj = -1e9
	best_soft_obj = -1e9
	constraints_satisfied = 0
	# Track hyperparams of best configs so frozen inference can be deferred.
	best_config_hparams: Optional[Dict] = None
	best_soft_config_hparams: Optional[Dict] = None

	tune_sub_df = non_frozen_df.iloc[tune_idx].reset_index(drop=True)

	for global_lambda in global_lambda_grid:
		for delta_scale in delta_scale_grid:
			for head_lambda in head_lambda_grid:
				for ties_density in ties_density_grid:
					for dare_density in dare_density_grid:
						config_tag = (
							f"gl{global_lambda:.3f}_ds{delta_scale:.2f}"
							f"_hl{head_lambda:.2f}_td{ties_density:.2f}_dd{dare_density:.2f}"
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
						)

						apply_adapter_params(model, merged_params)

						# Tune inference only — frozen deferred to best config after loop.
						merged_tune_pack = run_inference(
							model, tokenizer, tune_sub_df, batch_size
						)

						# Restore PL adapter weights immediately after inference.
						apply_adapter_params(model, pl_lora_params)

						m_thr, tune_m = tune_threshold(
							merged_tune_pack["score"], tune_labels, threshold_grid, mcc_floor
						)

						# Frozen metrics are NaN placeholders; filled in for best config only.
						row = {
							"repeat_id": repeat_id,
							"method": "ties_lora_b_only" if lora_b_only else "ties_dare_merge",
							"global_lambda": global_lambda,
							"delta_scale": delta_scale,
							"head_lambda": head_lambda,
							"ties_density": ties_density,
							"dare_density": dare_density,
							"threshold": m_thr,
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
							"frozen_routed_to_en_pct":       global_lambda,
							"frozen_mean_en_attn":           global_lambda,
							"frozen_en_win_precision_routed": float("nan"),
							"frozen_disagreement_acc":       float("nan"),
						}
						curve_rows.append(row)

						hparams = dict(
							global_lambda=global_lambda, delta_scale=delta_scale,
							head_lambda=head_lambda, ties_density=ties_density,
							dare_density=dare_density, config_tag=config_tag, threshold=m_thr,
						)

						# Selection objective: Macro-F1 on tune with MCC floor.
						if float(tune_m["mcc"]) + 1e-12 >= mcc_floor:
							constraints_satisfied += 1
							if float(tune_m["f1_macro"]) > best_obj:
								best_obj = float(tune_m["f1_macro"])
								best_row = row
								best_config_hparams = hparams
						# Soft fallback: penalise MCC shortfall.
						v_mcc = max(0.0, mcc_floor - float(tune_m["mcc"]))
						soft_obj = float(tune_m["f1_macro"]) - 5.0 * v_mcc
						if soft_obj > best_soft_obj:
							best_soft_obj = soft_obj
							best_soft_row = row
							best_soft_config_hparams = hparams

	# Fallback if no config satisfied MCC floor.
	used_soft_fallback = False
	if best_row is None:
		best_row = best_soft_row if best_soft_row is not None else (
			max(curve_rows, key=lambda r: r["tune_macro_f1"]) if curve_rows else None
		)
		best_config_hparams = best_soft_config_hparams
		used_soft_fallback = True

	# Deferred frozen inference: run only for the selected best config.
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
		)
		apply_adapter_params(model, best_merged)
		best_frozen_pack = run_inference(model, tokenizer, frozen_df, batch_size)
		apply_adapter_params(model, pl_lora_params)

		best_thr = best_config_hparams["threshold"]
		frozen_m = metrics_from_score(frozen_labels, best_frozen_pack["score"], best_thr)
		gap_red = (float(frozen_m["f1_macro"]) - pl_frozen_macro) / max(
			1e-8, oracle_frozen_macro - pl_frozen_macro
		)
		# Update best_row in-place (same dict object is already in curve_rows).
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
			# Schema-compat placeholders (no per-sample routing in weight merging).
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


def _fmt(v) -> str:
	if v is None or (isinstance(v, float) and np.isnan(v)):
		return "--"
	return f"{v:.4f}"


def export_latex_sweep(curve_df: pd.DataFrame, path: str) -> None:
	sweep_cols = (
		["repeat_id", "global_lambda", "delta_scale", "head_lambda",
		 "ties_density", "dare_density"]
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


# Final evaluation table column order: matches the user-specified header.
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

# t_{0.025, n-1} for 95% CI.  For n>=30 t≈1.960; for n=20 t=2.093.
import math as _math
def _t95(n: int) -> float:
	"""Approximate 95% CI t-critical value for df=n-1 using Wilson-Hilferty."""
	if n <= 1:
		return float("nan")
	df = n - 1
	# Approximation: t_{0.025,df} via normal quantile transform; good for df>=5.
	# Use lookup for small df, approximation for large.
	_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
			  6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
			  11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
			  16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
			  25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
	if df in _table:
		return _table[df]
	# Interpolate for unlisted df.
	keys = sorted(_table)
	for i in range(len(keys) - 1):
		if keys[i] < df < keys[i + 1]:
			t0, t1 = _table[keys[i]], _table[keys[i + 1]]
			a = (df - keys[i]) / (keys[i + 1] - keys[i])
			return t0 + a * (t1 - t0)
	return 1.960  # large df


def export_latex_final_eval(robust_df: pd.DataFrame, path: str) -> None:
	"""
	Export the final evaluation table with 95% CI for each metric.

	Output file: table_final_evaluation.tex
	Header: Method & Accuracy & Macro F1 & Weighted F1 & Precision (hate)
	        & Recall (hate) & F1 (hate) & PR AUC & MCC
	Cell format: mean (CI_lo--CI_hi)
	"""
	col_headers = [m[2] for m in FINAL_EVAL_METRICS]
	header_str = "Method & " + " & ".join(col_headers) + r" \\"
	n_cols = 1 + len(FINAL_EVAL_METRICS)
	lines = [
		r"\begin{table}[ht]", r"\centering",
		r"\caption{Final evaluation on the frozen held-out test set (" +
		r"mean with 95\% CI across repeated train/tune splits).}",
		r"\label{tab:final_evaluation}",
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

	threshold_grid = parse_threshold_grid(args.threshold_grid)
	temp_grid = [float(x) for x in args.temperature_grid.split(",")]
	global_lambda_grid = parse_float_grid(args.global_lambda_grid)
	delta_scale_grid   = parse_float_grid(args.delta_scale_grid)
	head_lambda_grid   = parse_float_grid(args.head_lambda_grid)
	ties_density_grid  = parse_float_grid(args.ties_density_grid)
	dare_density_grid  = parse_float_grid(args.dare_density_grid)

	# Load probe_acc_delta (layer-adaptive signal).
	probe_acc_delta = load_probe_acc_delta(args.stage2_per_layer_path)
	print(f"[info] probe_acc_delta loaded: {len(probe_acc_delta)} layers, "
		  f"max delta={np.max(probe_acc_delta):.4f} at layer {np.argmax(probe_acc_delta)}")

	# Load data.
	frozen_df = load_frozen_tuples(args.frozen_test_path)
	frozen_ids = set(frozen_df["id"].astype(str))
	non_frozen_df = load_non_frozen_polish_data(frozen_ids, hf_token=args.hf_token)
	print(f"[info] non_frozen={len(non_frozen_df)}, frozen={len(frozen_df)}")

	# ------------------------------------------------------------------ #
	# Step 1: Load EN model, extract EN lora params + EN predictions.    #
	# ------------------------------------------------------------------ #
	print("[info] Loading EN adapter model...")
	en_model, tokenizer = load_model(args.en_adapter_dir, hf_token=args.hf_token)

	print("[info] Running EN inference on full dataset...")
	en_full   = run_inference(en_model, tokenizer, non_frozen_df, args.batch_size)
	frozen_en = run_inference(en_model, tokenizer, frozen_df,     args.batch_size)

	en_lora_params = collect_adapter_params(en_model)
	print(f"[info] EN adapter params: {len(en_lora_params)} tensors")

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

	# Sanity check: PL and EN adapters have the same parameter keys.
	missing_in_en = set(pl_lora_params) - set(en_lora_params)
	if missing_in_en:
		print(f"[warn] {len(missing_in_en)} PL adapter keys not in EN: "
			  f"{list(missing_in_en)[:3]} ...")

	# Verify layer index parsing on a few keys (diagnostic).
	n_parsed = sum(1 for k in pl_lora_params if parse_layer_from_key(k) is not None and is_lora_key(k))
	n_lora_total = sum(1 for k in pl_lora_params if is_lora_key(k))
	print(f"[info] Layer index parseable: {n_parsed}/{n_lora_total} LoRA keys")

	# Count configs.
	n_configs = (len(global_lambda_grid) * len(delta_scale_grid)
				 * len(head_lambda_grid) * len(ties_density_grid)
				 * len(dare_density_grid))
	print(f"[info] Sweep: {n_configs} configs × {args.n_repeats} repeats = "
		  f"{n_configs * args.n_repeats} evaluations"
		  f"{' [lora_B-only merge]' if args.lora_b_only else ''}")

	# ------------------------------------------------------------------ #
	# Step 3: Repeat loop.                                                #
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
			probe_acc_delta=probe_acc_delta,
			mcc_floor_delta=args.mcc_floor_delta,
			tokenizer=tokenizer,
			batch_size=args.batch_size,
			lora_b_only=args.lora_b_only,
			val_size=args.val_size,
		)

		all_repeat_rows.extend(repeat_rows)
		all_curve_rows.extend(curve_rows)
		all_diagnostics.append(diagnostics)

		if repeat_rows:  # type: ignore[truthy-bool]
			best = repeat_rows[0]
			print(
				f"[repeat {rep}] best config: "
				f"gl={best['global_lambda']:.2f} ds={best['delta_scale']:.2f} "
				f"hl={best['head_lambda']:.2f} td={best['ties_density']:.2f} "
				f"dd={best['dare_density']:.2f}  "
				f"frozen_macro_f1={best['frozen_macro_f1']:.4f} "
				f"(PL={best['pl_frozen_macro_f1']:.4f})"
			)
		print(
			f"[repeat {rep}] PL={diagnostics['pl_frozen_macro_f1']:.4f} "
			f"oracle={diagnostics['oracle_frozen_macro_f1']:.4f} "
			f"constraints_ok={diagnostics['constraints_satisfied']}"
		)

	# ------------------------------------------------------------------ #
	# Step 4: Save results.                                               #
	# ------------------------------------------------------------------ #
	per_repeat_df = pd.DataFrame(all_repeat_rows)
	curve_df      = pd.DataFrame(all_curve_rows)
	diag_df       = pd.DataFrame(all_diagnostics)

	# wins_over_pl — derive method name from actual output rows for consistency.
	wins = 0
	method_name = "ties_lora_b_only" if args.lora_b_only else "ties_dare_merge"
	if "frozen_macro_f1" in per_repeat_df.columns and "pl_frozen_macro_f1" in per_repeat_df.columns:
		wins = int((per_repeat_df["frozen_macro_f1"] > per_repeat_df["pl_frozen_macro_f1"]).sum())
	wins_df = pd.DataFrame([{"method": method_name, "n_repeats_beat_pl": wins}])

	robust_df = aggregate_robustness(per_repeat_df) if len(per_repeat_df) > 0 else pd.DataFrame()

	# Write CSVs.
	per_repeat_df.to_csv(os.path.join(args.output_dir, "per_repeat_best.csv"), index=False)
	curve_df.to_csv(os.path.join(args.output_dir, "coverage_utility_curve.csv"), index=False)
	diag_df.to_csv(os.path.join(args.output_dir, "repeat_diagnostics.csv"), index=False)
	wins_df.to_csv(os.path.join(args.output_dir, "wins_over_pl.csv"), index=False)
	if len(robust_df) > 0:
		robust_df.to_csv(os.path.join(args.output_dir, "robustness_summary.csv"), index=False)

	# Write LaTeX tables.
	if len(curve_df) > 0:
		export_latex_sweep(
			curve_df, os.path.join(args.output_dir, "appendix_sweep.tex")
		)
	if len(robust_df) > 0:
		export_latex_robustness(
			robust_df, os.path.join(args.output_dir, "table_robustness.tex")
		)
		export_latex_final_eval(
			robust_df, os.path.join(args.output_dir, "table_final_evaluation.tex")
		)

	print("\n===== Summary =====")
	print(wins_df.to_string(index=False))
	if len(robust_df) > 0:
		print(robust_df[["method", "macro_f1_mean", "macro_f1_std",
						  "pl_frozen_macro_f1_mean", "oracle_frozen_macro_f1_mean",
						  "gap_reduction_mean"]].to_string(index=False))
	print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
	main()
