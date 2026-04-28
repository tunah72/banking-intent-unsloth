import os
import json
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
from unsloth import FastModel


def main(config_path):
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    checkpoint_dir = config['model_checkpoint']

    print("Loading label mapping...")
    with open(config['label_mapping_path'], 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    id2label = {int(k): v for k, v in mapping.items()}

    if not os.path.exists(checkpoint_dir):
        print(f"\n[ERROR] Model checkpoint not found at {checkpoint_dir}")
        print("Please run training first (bash train.sh).\n")
        exit(1)

    # Load fine-tuned model via Unsloth FastModel.
    # FastModel resolves the PEFT adapter → loads base model with Unsloth's
    # Triton kernels → restores LoRA adapters + score head (modules_to_save).
    print(f"Loading model via Unsloth FastModel from {checkpoint_dir}...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=checkpoint_dir,
        max_seq_length=config['max_seq_length'],
        dtype=None,
        load_in_4bit=True,
        num_labels=len(id2label),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Align modules_to_save (score head) dtype with backbone compute dtype.
    # 4-bit quantization keeps activations in float16/bfloat16, but the saved
    # classification head is restored as float32, causing a matmul dtype mismatch.
    compute_dtype = next(
        (p.dtype for p in model.parameters()
         if p.dtype in (torch.float16, torch.bfloat16)),
        torch.float16,
    )
    for module in model.modules():
        if hasattr(module, 'modules_to_save') and hasattr(module.modules_to_save, 'values'):
            for saved_module in module.modules_to_save.values():
                saved_module.to(compute_dtype)

    print(f"Loading test dataset from {config['test_data_path']}...")
    df_test = pd.read_csv(config['test_data_path'])

    os.makedirs(config['output_dir'], exist_ok=True)

    y_true_names = df_test['label_name'].tolist()
    y_pred_names = []

    device = next(model.parameters()).device

    print("Running batch inference on test dataset...")
    batch_size = 16
    for i in tqdm(range(0, len(df_test), batch_size)):
        batch_texts = df_test['text'].iloc[i:i + batch_size].tolist()
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=config['max_seq_length'],
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_class_ids = outputs.logits.argmax(-1).cpu().tolist()
        for class_id in predicted_class_ids:
            y_pred_names.append(id2label[class_id])

    # ── Save prediction CSV ────────────────────────────────────────────────
    df_test['predicted_label_name'] = y_pred_names
    pred_csv_path = os.path.join(config['output_dir'], "test_predictions.csv")
    df_test.to_csv(pred_csv_path, index=False)
    print(f"\n[1] Saved detailed predictions to: {pred_csv_path}")

    # ── Metrics ────────────────────────────────────────────────────────────
    accuracy     = accuracy_score(y_true_names, y_pred_names)
    target_names = [id2label[i] for i in range(len(id2label))]
    report_dict  = classification_report(
        y_true_names, y_pred_names, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    report_text  = classification_report(
        y_true_names, y_pred_names, target_names=target_names, zero_division=0,
    )

    report_txt_path = os.path.join(config['output_dir'], "classification_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[2] Saved classification report to: {report_txt_path}")

    metrics_json_path = os.path.join(config['output_dir'], "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)
    print(f"[3] Saved metrics JSON to: {metrics_json_path}")

    # ── Confusion Matrix ───────────────────────────────────────────────────
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true_names, y_pred_names, labels=target_names)
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Banking77 Intent Classification - Confusion Matrix", fontsize=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_path = os.path.join(config['output_dir'], "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[4] Saved confusion matrix to: {cm_path}")

    print("\n" + "=" * 50)
    print(" EVALUATION COMPLETED SUCCESSFULLY")
    print(f" Overall Accuracy: {accuracy*100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/evaluate.yaml")
    args = parser.parse_args()
    main(args.config)
