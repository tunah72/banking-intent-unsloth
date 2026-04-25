import os
import yaml
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
from unsloth import FastSequenceClassificationModel
from transformers import TrainingArguments, Trainer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_dataset(csv_path, tokenizer, max_length):
    df = pd.read_csv(csv_path)
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    return tokenized_dataset


def main(config_path):
    config = load_config(config_path)

    # ── Model & Tokenizer (Unsloth) ─────────────────────────────────────────
    print(f"Loading model with Unsloth: {config['model_name']}...")
    model, tokenizer = FastSequenceClassificationModel.from_pretrained(
        model_name=config['model_name'],
        num_labels=config['num_labels'],
        max_seq_length=config['max_seq_length'],
        dtype=None,       # auto: float16 on T4, bfloat16 on Ampere+
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # ── LoRA (Unsloth) ──────────────────────────────────────────────────────
    print("Applying LoRA adapters...")
    model = FastSequenceClassificationModel.get_peft_model(
        model,
        r=config['lora_r'],
        target_modules=config['target_modules'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        bias="none",
        use_gradient_checkpointing=config.get('use_gradient_checkpointing', 'unsloth'),
        random_state=config['training_args']['seed'],
    )
    model.print_trainable_parameters()

    # ── Datasets ────────────────────────────────────────────────────────────
    print("Loading and tokenizing datasets...")
    train_dataset = prepare_dataset("sample_data/train.csv", tokenizer, config['max_seq_length'])
    val_dataset   = prepare_dataset("sample_data/val.csv",   tokenizer, config['max_seq_length'])

    # ── Training Arguments ──────────────────────────────────────────────────
    t_args = config['training_args']
    training_args = TrainingArguments(
        output_dir=t_args['output_dir'],
        per_device_train_batch_size=t_args['per_device_train_batch_size'],
        gradient_accumulation_steps=t_args['gradient_accumulation_steps'],
        learning_rate=t_args['learning_rate'],
        num_train_epochs=t_args['num_train_epochs'],
        eval_strategy=t_args['eval_strategy'],
        eval_steps=t_args['eval_steps'],
        save_strategy=t_args['save_strategy'],
        save_steps=t_args['save_steps'],
        logging_steps=t_args['logging_steps'],
        save_total_limit=t_args['save_total_limit'],
        load_best_model_at_end=t_args['load_best_model_at_end'],
        metric_for_best_model=t_args['metric_for_best_model'],
        greater_is_better=t_args['greater_is_better'],
        optim=t_args['optim'],
        weight_decay=t_args['weight_decay'],
        lr_scheduler_type=t_args['lr_scheduler_type'],
        warmup_ratio=t_args['warmup_ratio'],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        seed=t_args['seed'],
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Resume from checkpoint if one exists (handles Kaggle/Colab disconnection)
    output_dir = t_args['output_dir']
    resume_from_checkpoint = False
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            resume_from_checkpoint = True
            print(f"Found existing checkpoints in {output_dir}. Resuming training...")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print("Training complete! Saving final best model...")
    final_save_dir = os.path.join(output_dir, "final_best_model")
    model.save_pretrained(final_save_dir)
    model.config.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print(f"Best model saved to {final_save_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()
    main(args.config)
