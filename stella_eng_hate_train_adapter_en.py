import gc
import importlib.util
import inspect
import os
import numpy as np
import pandas as pd
import torch
import evaluate
from datasets import ClassLabel, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import login
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

# =========================
# Config
# =========================

HF_TOKEN = os.environ["HF_TOKEN"]
DATASET_ID = "manueltonneau/english-hate-speech-superset"
MODEL_ID = "sdadas/stella-pl"
OUTPUT_DIR = "stella_pl_hate_lora"
MAX_LENGTH = 256
TEST_SIZE = 10_000
VAL_SIZE = 10_000
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
TRANSFER_SEED = FINAL_TRAIN_SEED

if not torch.cuda.is_available():
    raise RuntimeError("This script targets an NVIDIA GPU (e.g., L4) with CUDA.")

torch.backends.cuda.matmul.allow_tf32 = True
print("CUDA device:", torch.cuda.get_device_name(0))

HAS_FLASH_ATTN = importlib.util.find_spec("flash_attn") is not None
ATTN_IMPLEMENTATION = "flash_attention_2" if HAS_FLASH_ATTN else "sdpa"
print("Attention implementation:", ATTN_IMPLEMENTATION)

# =========================
# Auth + dataset load
# =========================
login(token=HF_TOKEN)
dataset = load_dataset(DATASET_ID, token=HF_TOKEN)


def find_text_column(ds):
    preferred = ["text", "sentence", "content", "comment", "post", "tweet"]

    def is_string_feature(col_name):
        feature = ds.features.get(col_name)
        return feature is not None and getattr(feature, "dtype", None) == "string"

    for col in preferred:
        if col in ds.column_names and is_string_feature(col):
            return col

    # Fallback: any column with string dtype except labels.
    for col in ds.column_names:
        if col != "labels" and is_string_feature(col):
            return col

    # Last resort: choose a non-label column with the highest proportion of string-like values.
    best_col = None
    best_score = -1.0
    for col in ds.column_names:
        if col == "labels":
            continue
        sample = ds.select(range(min(200, len(ds))))[col]
        if len(sample) == 0:
            continue
        stringish = sum(
            1
            for value in sample
            if value is None or isinstance(value, str) or isinstance(value, (list, tuple))
        )
        score = stringish / len(sample)
        if score > best_score:
            best_score = score
            best_col = col

    if best_col is not None:
        return best_col
    raise ValueError("Could not find a text column.")


def datasetdict_to_single_dataset(dataset_dict):
    split_names = list(dataset_dict.keys())
    if len(split_names) == 1:
        return dataset_dict[split_names[0]]
    return concatenate_datasets([dataset_dict[name] for name in split_names])


def cast_labels(batch):
    return {"labels": [int(value) for value in batch["labels"]]}


# =========================
# 1) Frozen test set
# =========================
full_dataset = datasetdict_to_single_dataset(dataset).map(cast_labels, batched=True)

# Stratified splits in `datasets` require ClassLabel, so remap labels to contiguous ids.
raw_label_values = sorted(set(full_dataset["labels"]))
label_to_id = {label_value: idx for idx, label_value in enumerate(raw_label_values)}

def remap_labels(batch):
    return {"labels": [label_to_id[value] for value in batch["labels"]]}

full_dataset = full_dataset.map(remap_labels, batched=True)
full_dataset = full_dataset.cast_column(
    "labels",
    ClassLabel(names=[str(value) for value in raw_label_values]),
)

frozen_split = full_dataset.train_test_split(
    test_size=TEST_SIZE,
    seed=TEST_SEED,
    stratify_by_column="labels",
)
dev_pool = frozen_split["train"]
frozen_test_ds = frozen_split["test"]

text_col = find_text_column(dev_pool)
num_labels = len(sorted(set(dev_pool["labels"])))

print("Text column:", text_col)
print("Num labels:", num_labels)
print("Development pool size:", len(dev_pool))
print("Frozen test size:", len(frozen_test_ds))

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
    return DatasetDict({
        "train": split["train"],
        "validation": split["test"],
        "test": frozen_test_ds,
    })


def preprocess(batch):
    values = batch[text_col]
    normalized_texts = []
    for value in values:
        if value is None:
            normalized_texts.append("")
        elif isinstance(value, str):
            normalized_texts.append(value)
        elif isinstance(value, (list, tuple)):
            normalized_texts.append(" ".join(str(item) for item in value))
        else:
            normalized_texts.append(str(value))
    return tokenizer(normalized_texts, truncation=True, max_length=MAX_LENGTH)


def tokenize_splits(splits):
    tokenized = splits.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("labels", "label")

    keep_cols = {"input_ids", "attention_mask", "label"}
    for split_name in tokenized.keys():
        remove_cols = [c for c in tokenized[split_name].column_names if c not in keep_cols]
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
        num_labels=num_labels,
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
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
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

    # Transformers API changed from evaluation_strategy -> eval_strategy.
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
        num_labels=num_labels,
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
# 4) Development study (3x2 = 6 runs)
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

print("Development runs complete.")

# =========================
# 5) Final locked evaluation
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
final_test_logits = final_test_output.predictions[0] if isinstance(final_test_output.predictions, tuple) else final_test_output.predictions
final_test_labels = final_test_output.label_ids.astype(int)

pd.DataFrame([test_metrics]).to_csv(os.path.join(final_run_dir, "frozen_test_metrics.csv"), index=False)

adapter_dir = os.path.join(final_run_dir, "best_adapter")
best_model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

print("Final frozen-test evaluation complete.")

# =========================
# 6) Bootstrap 95% CI + LaTeX tables
# =========================
REPORT_METRICS = [
    ("test_accuracy", "Accuracy"),
    ("test_f1_macro", "Macro F1"),
    ("test_f1_weighted", "Weighted F1"),
    ("test_precision_pos", "Precision (positive)"),
    ("test_recall_pos", "Recall (positive)"),
    ("test_f1_pos", "F1 (positive)"),
    ("test_pr_auc", "PR-AUC"),
]

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
ci_rows = []
for metric_key, metric_label in REPORT_METRICS:
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

final_test_table_df = pd.DataFrame(ci_rows)
latex_test_metrics_table = final_test_table_df[["metric", "latex_value"]].rename(
    columns={"metric": "Metric", "latex_value": "Value"}
).to_latex(index=False, escape=False)
latex_test_ci_table = final_test_table_df[["metric", "latex_value_pm_ci"]].rename(
    columns={"metric": "Metric", "latex_value_pm_ci": "Value $\\pm$ CI interval"}
).to_latex(index=False, escape=False)

with open(os.path.join(final_run_dir, "table_frozen_test_metrics.tex"), "w", encoding="utf-8") as f:
    f.write(latex_test_metrics_table)
with open(os.path.join(final_run_dir, "table_frozen_test_bootstrap_ci.tex"), "w", encoding="utf-8") as f:
    f.write(latex_test_ci_table)

final_test_table_df.to_csv(os.path.join(final_run_dir, "frozen_test_metrics_with_ci.csv"), index=False)
print("Bootstrap CI and LaTeX tables exported.")

# =========================
# 7) Train on all non-test data and export transfer checkpoint
# =========================
transfer_run_dir = os.path.join(OUTPUT_DIR, "transfer_ready")
transfer_eval_size = min(5000, max(2000, int(0.02 * len(dev_pool))))

transfer_split = dev_pool.train_test_split(
    test_size=transfer_eval_size,
    seed=TRANSFER_SEED,
    stratify_by_column="labels",
)
transfer_splits = DatasetDict({
    "train": transfer_split["train"],
    "validation": transfer_split["test"],
})
transfer_tokenized = tokenize_splits(transfer_splits)
transfer_model = build_model()

transfer_trainer = Trainer(
    model=transfer_model,
    args=build_training_args(transfer_run_dir, TRANSFER_SEED),
    train_dataset=transfer_tokenized["train"],
    eval_dataset=transfer_tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

transfer_trainer.train()
transfer_metrics = transfer_trainer.evaluate(transfer_tokenized["validation"], metric_key_prefix="transfer_validation")
transfer_best_checkpoint = transfer_trainer.state.best_model_checkpoint
if transfer_best_checkpoint is None:
    raise RuntimeError("No best checkpoint was recorded during transfer-stage training.")

transfer_export_model = build_inference_model(transfer_best_checkpoint)
transfer_adapter_dir = os.path.join(transfer_run_dir, "best_adapter")
transfer_export_model.save_pretrained(transfer_adapter_dir)
tokenizer.save_pretrained(transfer_adapter_dir)

pd.DataFrame([transfer_metrics]).to_csv(os.path.join(transfer_run_dir, "transfer_validation_metrics.csv"), index=False)
print("Transfer checkpoint ready:", transfer_adapter_dir)
print("Done. Frozen test set was never used for training in the transfer stage.")