"""
Stage A — Fine-tune a small seq2seq model to decompose a passage into atomic claims.

This is the model that actually ships in Veritas's verifier. It runs standalone
at inference time (see inference.py) — no external LLM API calls.

Input format:  "decompose: <passage>"
Target format: "<claim 1> <SEP> <claim 2> <SEP> <claim 3> ..."

Usage
-----
python train_stage_a.py \
    --data_file claims_dataset.jsonl \
    --base_model google/flan-t5-small \
    --output_dir ./stage_a_model \
    --epochs 6
"""

import argparse
import json

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

CLAIM_SEP = " <CLAIM_SEP> "
INPUT_PREFIX = "decompose: "

MAX_INPUT_LEN = 256
MAX_TARGET_LEN = 256


def load_jsonl(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("passage") and obj.get("claims"):
                examples.append(obj)
    return examples


def to_hf_dataset(examples, tokenizer):
    inputs = [INPUT_PREFIX + ex["passage"] for ex in examples]
    targets = [CLAIM_SEP.join(ex["claims"]) for ex in examples]

    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LEN, truncation=True)
    labels = tokenizer(text_target=targets, max_length=MAX_TARGET_LEN, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return Dataset.from_dict(model_inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="google/flan-t5-small",
                         help="google/flan-t5-small (~80M params, fast) or google/flan-t5-base (~250M, better quality)")
    parser.add_argument("--output_dir", type=str, default="./stage_a_model")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading data from {args.data_file} ...")
    examples = load_jsonl(args.data_file)
    print(f"Loaded {len(examples)} labeled examples.")
    if len(examples) < 50:
        print("WARNING: fewer than 50 examples. Consider generating more silver "
              "labels with data_generation.py before training — Stage A quality "
              "depends heavily on dataset size and diversity.")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(examples))
    n_val = max(1, int(len(examples) * args.val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_examples = [examples[i] for i in train_idx]
    val_examples = [examples[i] for i in val_idx]
    print(f"Train: {len(train_examples)}  Val: {len(val_examples)}")

    print(f"Loading base model {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    train_ds = to_hf_dataset(train_examples, tokenizer)
    val_ds = to_hf_dataset(val_examples, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        logging_steps=10,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    print(f"Saving final model to {args.output_dir}/final")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    # Quick qualitative check on a couple of validation examples
    print("\n--- Sample predictions ---")
    for ex in val_examples[:3]:
        input_text = INPUT_PREFIX + ex["passage"]
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN).input_ids
        output_ids = model.generate(input_ids, max_length=MAX_TARGET_LEN, num_beams=4)
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nPassage: {ex['passage'][:200]}...")
        print(f"Gold claims: {ex['claims']}")
        print(f"Predicted:   {pred.split(CLAIM_SEP.strip())}")


if __name__ == "__main__":
    main()
