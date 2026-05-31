import gc
import importlib.util
import inspect
import json
import os

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	BitsAndBytesConfig,
	DataCollatorWithPadding,
	EarlyStoppingCallback,
	Trainer,
	TrainingArguments,
	set_seed,
)

# =========================
# Config
# =========================

HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET_ID = "community-datasets/hate_speech_pl"
MODEL_ID = "sdadas/stella-pl"
OUTPUT_DIR = "stella_pl_hate_lora_hatespeech_pl_strict"
MAX_LENGTH = 256

# Strict threshold: rating 0 -> 0, rating 1-4 -> 1
STRICT_LABEL_MAP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}

# Small-dataset split strategy agreed for n=13,887
TEST_SIZE = 2777
VAL_SIZE = 2222
TEST_SEED = 42
SPLIT_SEEDS = [101, 202, 303]
TRAIN_SEEDS = [11, 22]
FINAL_SPLIT_SEED = SPLIT_SEEDS[0]
FINAL_TRAIN_SEED = TRAIN_SEEDS[0]

EARLY_STOPPING_PATIENCE = 2
NUM_EPOCHS = 5
LEARNING_RATE = 2e-4

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 123

if not torch.cuda.is_available():
	raise RuntimeError("This script targets an NVIDIA GPU (e.g., L4) with CUDA.")

torch.backends.cuda.matmul.allow_tf32 = True
print("CUDA device:", torch.cuda.get_device_name(0))

HAS_FLASH_ATTN = importlib.util.find_spec("flash_attn") is not None
ATTN_IMPLEMENTATION = "flash_attention_2" if HAS_FLASH_ATTN else "sdpa"
print("Attention implementation:", ATTN_IMPLEMENTATION)


# =========================
# Dataset helpers
# =========================
def clean_html_tags(text):
	if not isinstance(text, str):
		return text
	import re

	clean = re.sub(r"<[^>]+>", "", text)
	clean = re.sub(r"\s+", " ", clean).strip()
	return clean


def datasetdict_to_single_dataset(dataset_obj):
	if isinstance(dataset_obj, Dataset):
		return dataset_obj
	split_names = list(dataset_obj.keys())
	if len(split_names) == 1:
		return dataset_obj[split_names[0]]
	return concatenate_datasets([dataset_obj[name] for name in split_names])


def find_text_column(ds):
	preferred = ["text", "sentence", "content", "comment", "post", "tweet", "test_case"]

	def is_string_feature(col_name):
		feature = ds.features.get(col_name)
		return feature is not None and getattr(feature, "dtype", None) == "string"

	for col in preferred:
		if col in ds.column_names and is_string_feature(col):
			return col

	for col in ds.column_names:
		if col in {"labels", "label", "label_int", "rating"}:
			continue
		if is_string_feature(col):
			return col

	raise ValueError(f"Could not find a text column in: {ds.column_names}")


def find_id_column(ds):
	preferred = ["id", "ID", "comment_id", "uid", "item_id", "doc_id"]
	for col in preferred:
		if col in ds.column_names:
			return col
	return None


def add_row_id_if_missing(ds, id_col):
	if id_col is not None:
		return ds, id_col
	ds = ds.add_column("row_id", list(range(len(ds))))
	return ds, "row_id"


def to_strict_label(value):
	if value is None:
		return None
	try:
		numeric = int(value)
	except (TypeError, ValueError):
		return None
	return STRICT_LABEL_MAP.get(numeric)


def preprocess_and_label(batch, text_col):
	texts = []
	labels = []
	keep = []

	for idx, raw_rating in enumerate(batch["rating"]):
		mapped = to_strict_label(raw_rating)
		if mapped is None:
			keep.append(False)
			texts.append(None)
			labels.append(None)
			continue

		text_val = batch[text_col][idx]
		if text_val is None:
			text_val = ""
		elif not isinstance(text_val, str):
			text_val = str(text_val)

		texts.append(clean_html_tags(text_val))
		labels.append(mapped)
		keep.append(True)

	return {
		"text": texts,
		"labels": labels,
		"_keep": keep,
	}


def bootstrap_ci_table(test_metrics, final_test_logits, final_test_labels):
	rng = np.random.default_rng(BOOTSTRAP_SEED)
	n_test = len(final_test_labels)
	bootstrap_records = []
	for _ in range(BOOTSTRAP_SAMPLES):
		sample_idx = rng.integers(0, n_test, size=n_test)
		sampled_logits = final_test_logits[sample_idx]
		sampled_labels = final_test_labels[sample_idx]
		sampled_metrics = compute_metrics((sampled_logits, sampled_labels))
		bootstrap_records.append({f"test_{key}": value for key, value in sampled_metrics.items()})

	bootstrap_df = pd.DataFrame(bootstrap_records)
	report_metrics = [
		("test_accuracy", "Accuracy"),
		("test_f1_macro", "Macro F1"),
		("test_f1_weighted", "Weighted F1"),
		("test_precision_pos", "Precision (positive)"),
		("test_recall_pos", "Recall (positive)"),
		("test_f1_pos", "F1 (positive)"),
		("test_pr_auc", "PR-AUC"),
	]

	ci_rows = []
	for metric_key, metric_label in report_metrics:
		point_estimate = float(test_metrics[metric_key])
		lower = float(bootstrap_df[metric_key].quantile(0.025))
		upper = float(bootstrap_df[metric_key].quantile(0.975))
		half_width = (upper - lower) / 2.0
		ci_rows.append(
			{
				"metric": metric_label,
				"value": point_estimate,
				"ci_lower": lower,
				"ci_upper": upper,
				"ci_half_width": half_width,
				"latex_value": f"{point_estimate:.2f}",
				"latex_value_pm_ci": f"{point_estimate:.2f} \\pm {half_width:.2f}",
			}
		)

	return pd.DataFrame(ci_rows)


def export_stability_summary_latex(stability_summary_df, output_dir):
	latex_ready = stability_summary_df.reset_index().rename(
		columns={"index": "Metric", "mean": "Mean", "std": "Std", "se": "SE"}
	)
	for col in ["Mean", "Std", "SE"]:
		latex_ready[col] = latex_ready[col].map(lambda x: f"{float(x):.4f}")

	latex_table = latex_ready[["Metric", "Mean", "Std", "SE"]].to_latex(index=False, escape=False)
	tex_path = os.path.join(output_dir, "table_development_stability_summary.tex")
	with open(tex_path, "w", encoding="utf-8") as f:
		f.write(latex_table)
	print("Stability LaTeX table exported:", tex_path)


# =========================
# Auth + load dataset
# =========================
load_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
dataset = load_dataset(DATASET_ID, **load_kwargs)
full_dataset = datasetdict_to_single_dataset(dataset)

raw_text_col = find_text_column(full_dataset)
raw_id_col = find_id_column(full_dataset)
full_dataset, id_col = add_row_id_if_missing(full_dataset, raw_id_col)

if "rating" not in full_dataset.column_names:
	raise ValueError("Expected `rating` column in community-datasets/hate_speech_pl.")

full_dataset = full_dataset.map(
	lambda b: preprocess_and_label(b, raw_text_col),
	batched=True,
)
full_dataset = full_dataset.filter(lambda x: x["_keep"])
full_dataset = full_dataset.remove_columns(["_keep"])

# Keep only data required for reproducible strict-threshold experiments.
keep_cols = {"text", "labels", id_col}
drop_cols = [c for c in full_dataset.column_names if c not in keep_cols]
if drop_cols:
	full_dataset = full_dataset.remove_columns(drop_cols)

full_dataset = full_dataset.cast_column("labels", ClassLabel(names=["non-hateful", "hateful"]))

print("Dataset size after strict-threshold preprocessing:", len(full_dataset))
print("Text column:", "text")
print("ID column:", id_col)
print("Label distribution (strict):")
print(pd.Series(full_dataset["labels"]).value_counts().sort_index().to_string())


# =========================
# 1) Frozen held-out test set
# =========================
frozen_split = full_dataset.train_test_split(
	test_size=TEST_SIZE,
	seed=TEST_SEED,
	stratify_by_column="labels",
)
dev_pool = frozen_split["train"]
frozen_test_ds = frozen_split["test"]

print("Development pool size:", len(dev_pool))
print("Frozen test size:", len(frozen_test_ds))


def serialize_frozen_test(ds, output_dir, id_column):
	os.makedirs(output_dir, exist_ok=True)

	records_df = pd.DataFrame(
		{
			"label_strict": [int(v) for v in ds["labels"]],
			"text": ds["text"],
			"id": [str(v) for v in ds[id_column]],
		}
	)

	csv_path = os.path.join(output_dir, "frozen_test_strict_records.csv")
	jsonl_path = os.path.join(output_dir, "frozen_test_strict_tuples.jsonl")

	records_df.to_csv(csv_path, index=False)
	with open(jsonl_path, "w", encoding="utf-8") as f:
		for row in records_df.itertuples(index=False):
			# Explicit tuple serialization: [label_strict, text, id]
			f.write(json.dumps([int(row.label_strict), row.text, row.id], ensure_ascii=False) + "\n")

	print("Frozen test serialized:", csv_path)
	print("Frozen test tuples serialized:", jsonl_path)


serialize_frozen_test(frozen_test_ds, OUTPUT_DIR, id_col)


# =========================
# 2) Tokenization + metrics
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
	tokenizer.pad_token = tokenizer.eos_token

collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def make_dev_splits(split_seed):
	split = dev_pool.train_test_split(
		test_size=VAL_SIZE,
		seed=split_seed,
		stratify_by_column="labels",
	)
	return DatasetDict(
		{
			"train": split["train"],
			"validation": split["test"],
			"test": frozen_test_ds,
		}
	)


def preprocess(batch):
	return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)


def tokenize_splits(splits):
	tokenized = splits.map(preprocess, batched=True)
	tokenized = tokenized.rename_column("labels", "label")

	keep_cols_local = {"input_ids", "attention_mask", "label"}
	for split_name in tokenized.keys():
		remove_cols = [c for c in tokenized[split_name].column_names if c not in keep_cols_local]
		tokenized[split_name] = tokenized[split_name].remove_columns(remove_cols)

	tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
	return tokenized


def compute_metrics(eval_pred):
	logits, labels = eval_pred
	logits = logits[0] if isinstance(logits, tuple) else logits
	labels = labels.astype(int)
	preds = np.argmax(logits, axis=-1)

	shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
	probs = np.exp(shifted_logits)
	probs = probs / np.sum(probs, axis=-1, keepdims=True)
	pos_probs = probs[:, 1]

	pos_precision, pos_recall, pos_f1, _ = precision_recall_fscore_support(
		labels,
		preds,
		average="binary",
		pos_label=1,
		zero_division=0,
	)

	return {
		"accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
		"f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
		"f1_weighted": f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"],
		"precision_pos": pos_precision,
		"recall_pos": pos_recall,
		"f1_pos": pos_f1,
		"pr_auc": average_precision_score(labels, pos_probs),
	}


# =========================
# 3) Model helpers
# =========================
bnb_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_quant_type="nf4",
	bnb_4bit_use_double_quant=True,
	bnb_4bit_compute_dtype=torch.bfloat16,
)


def build_model():
	model = AutoModelForSequenceClassification.from_pretrained(
		MODEL_ID,
		trust_remote_code=True,
		num_labels=2,
		quantization_config=bnb_config,
		torch_dtype=torch.bfloat16,
		attn_implementation=ATTN_IMPLEMENTATION,
		device_map="auto",
	)
	model.config.pad_token_id = tokenizer.pad_token_id
	model.config.problem_type = "single_label_classification"
	model = prepare_model_for_kbit_training(model)

	lora_config = LoraConfig(
		r=16,
		lora_alpha=32,
		lora_dropout=0.05,
		bias="none",
		task_type="SEQ_CLS",
		target_modules=[
			"q_proj",
			"k_proj",
			"v_proj",
			"o_proj",
			"up_proj",
			"down_proj",
			"gate_proj",
		],
		modules_to_save=["score", "classifier"],
	)
	return get_peft_model(model, lora_config)


def build_training_args(output_dir, train_seed):
	training_kwargs = {
		"output_dir": output_dir,
		"overwrite_output_dir": True,
		"num_train_epochs": NUM_EPOCHS,
		"per_device_train_batch_size": 8,
		"per_device_eval_batch_size": 16,
		"gradient_accumulation_steps": 4,
		"learning_rate": LEARNING_RATE,
		"lr_scheduler_type": "cosine",
		"warmup_ratio": 0.05,
		"weight_decay": 0.01,
		"save_strategy": "epoch",
		"save_total_limit": 2,
		"logging_steps": 25,
		"bf16": True,
		"tf32": True,
		"gradient_checkpointing": True,
		"optim": "paged_adamw_8bit",
		"max_grad_norm": 0.3,
		"load_best_model_at_end": True,
		"metric_for_best_model": "f1_macro",
		"greater_is_better": True,
		"report_to": "none",
		"seed": train_seed,
		"data_seed": train_seed,
	}

	training_args_params = inspect.signature(TrainingArguments.__init__).parameters
	if "evaluation_strategy" in training_args_params:
		training_kwargs["evaluation_strategy"] = "epoch"
	elif "eval_strategy" in training_args_params:
		training_kwargs["eval_strategy"] = "epoch"
	else:
		raise RuntimeError("Neither evaluation_strategy nor eval_strategy is supported by this transformers version.")

	return TrainingArguments(**training_kwargs)


def cleanup_memory():
	gc.collect()
	torch.cuda.empty_cache()


def build_inference_model(adapter_path):
	base_model = AutoModelForSequenceClassification.from_pretrained(
		MODEL_ID,
		trust_remote_code=True,
		num_labels=2,
		quantization_config=bnb_config,
		torch_dtype=torch.bfloat16,
		attn_implementation=ATTN_IMPLEMENTATION,
		device_map="auto",
	)
	base_model.config.pad_token_id = tokenizer.pad_token_id
	base_model.config.problem_type = "single_label_classification"
	model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
	model.eval()
	return model


# =========================
# 4) Development stability study (3x2 = 6 runs)
# =========================
def run_development_experiment(split_seed, train_seed):
	run_name = f"split_{split_seed}_train_{train_seed}"
	run_dir = os.path.join(OUTPUT_DIR, "development", run_name)

	set_seed(train_seed)
	splits = make_dev_splits(split_seed)
	tokenized = tokenize_splits(splits)
	model = build_model()
	trainer = Trainer(
		model=model,
		args=build_training_args(run_dir, train_seed),
		train_dataset=tokenized["train"],
		eval_dataset=tokenized["validation"],
		tokenizer=tokenizer,
		data_collator=collator,
		compute_metrics=compute_metrics,
		callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
	)

	trainer.train()
	val_metrics = trainer.evaluate(tokenized["validation"], metric_key_prefix="validation")

	result = {
		"split_seed": split_seed,
		"train_seed": train_seed,
		"best_checkpoint": trainer.state.best_model_checkpoint,
		"best_metric": trainer.state.best_metric,
	}
	result.update(val_metrics)

	del trainer
	del model
	cleanup_memory()
	return result


development_results = []
for split_seed in SPLIT_SEEDS:
	for train_seed in TRAIN_SEEDS:
		print(f"Running development experiment split_seed={split_seed}, train_seed={train_seed}")
		development_results.append(run_development_experiment(split_seed, train_seed))

development_results_df = pd.DataFrame(development_results).sort_values(["split_seed", "train_seed"])
metric_cols = [
	"validation_loss",
	"validation_accuracy",
	"validation_f1_macro",
	"validation_f1_weighted",
	"validation_precision_pos",
	"validation_recall_pos",
	"validation_f1_pos",
	"validation_pr_auc",
]
stability_summary_df = development_results_df[metric_cols].agg(["mean", "std"]).T
stability_summary_df["se"] = stability_summary_df["std"] / np.sqrt(len(development_results_df))

os.makedirs(OUTPUT_DIR, exist_ok=True)
development_results_df.to_csv(os.path.join(OUTPUT_DIR, "development_results.csv"), index=False)
stability_summary_df.to_csv(os.path.join(OUTPUT_DIR, "development_stability_summary.csv"))
export_stability_summary_latex(stability_summary_df, OUTPUT_DIR)

print("Development runs complete.")


# =========================
# 5) Final locked evaluation (single pre-specified run)
# =========================
final_run_dir = os.path.join(OUTPUT_DIR, "final_locked_eval")
set_seed(FINAL_TRAIN_SEED)
final_splits = make_dev_splits(FINAL_SPLIT_SEED)
final_tokenized = tokenize_splits(final_splits)
final_model = build_model()

final_trainer = Trainer(
	model=final_model,
	args=build_training_args(final_run_dir, FINAL_TRAIN_SEED),
	train_dataset=final_tokenized["train"],
	eval_dataset=final_tokenized["validation"],
	tokenizer=tokenizer,
	data_collator=collator,
	compute_metrics=compute_metrics,
	callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

final_trainer.train()
best_checkpoint = final_trainer.state.best_model_checkpoint
if best_checkpoint is None:
	raise RuntimeError("No best checkpoint was recorded during final training.")

best_model = build_inference_model(best_checkpoint)
inference_trainer = Trainer(
	model=best_model,
	tokenizer=tokenizer,
	data_collator=collator,
	compute_metrics=compute_metrics,
)

final_test_output = inference_trainer.predict(final_tokenized["test"], metric_key_prefix="test")
test_metrics = final_test_output.metrics
final_test_logits = (
	final_test_output.predictions[0]
	if isinstance(final_test_output.predictions, tuple)
	else final_test_output.predictions
)
final_test_labels = final_test_output.label_ids.astype(int)

os.makedirs(final_run_dir, exist_ok=True)
pd.DataFrame([test_metrics]).to_csv(os.path.join(final_run_dir, "frozen_test_metrics.csv"), index=False)

adapter_dir = os.path.join(final_run_dir, "best_adapter")
best_model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

print("Final frozen-test evaluation complete.")
print("Best checkpoint selected by validation macro-F1:", best_checkpoint)


# =========================
# 6) Bootstrap 95% CI + LaTeX table
# =========================
final_test_table_df = bootstrap_ci_table(test_metrics, final_test_logits, final_test_labels)

latex_test_ci_table = final_test_table_df[["metric", "latex_value_pm_ci"]].rename(
	columns={"metric": "Metric", "latex_value_pm_ci": "Value $\\pm$ CI interval"}
).to_latex(index=False, escape=False)

with open(os.path.join(final_run_dir, "table_frozen_test_bootstrap_ci.tex"), "w", encoding="utf-8") as f:
	f.write(latex_test_ci_table)

final_test_table_df.to_csv(os.path.join(final_run_dir, "frozen_test_metrics_with_ci.csv"), index=False)
print("Bootstrap CI and LaTeX table exported.")
print("Done. Frozen test set was not used for model selection or training.")
