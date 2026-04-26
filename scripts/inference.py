import os
import json
import yaml
import torch
from unsloth import FastModel


class IntentClassification:
    def __init__(self, model_path):
        """
        Args:
            model_path: Path to the inference YAML config (e.g. configs/inference.yaml).
                        The config contains the actual checkpoint path under 'model_checkpoint'.
        """
        print(f"Loading configuration from {model_path}...")
        with open(model_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        checkpoint_dir = self.config['model_checkpoint']

        print("Loading label mapping...")
        with open(self.config['label_mapping_path'], 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        self.id2label = {int(k): v for k, v in mapping.items()}

        # Load fine-tuned model via Unsloth FastModel.
        # Passing checkpoint_dir (PEFT adapter path) causes FastModel to:
        #   1. Read adapter_config.json → resolve base model name
        #   2. Load base model with Unsloth's Triton kernels
        #   3. Apply saved LoRA adapters + restore score head (modules_to_save)
        print(f"Loading model via Unsloth FastModel from {checkpoint_dir}...")
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=checkpoint_dir,
            max_seq_length=self.config['max_seq_length'],
            dtype=None,
            load_in_4bit=True,
            num_labels=len(self.id2label),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        print("Model ready.")

    def __call__(self, message):
        """
        Args:
            message: Customer query string.
        Returns:
            Predicted intent label name (string).
        """
        inputs = self.tokenizer(
            message,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['max_seq_length'],
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_class_id = outputs.logits.argmax(-1).item()
        return self.id2label[predicted_class_id]


if __name__ == '__main__':
    config_file = "configs/inference.yaml"

    with open(config_file, 'r', encoding='utf-8') as f:
        _config = yaml.safe_load(f)

    if not os.path.exists(_config['model_checkpoint']):
        print(f"\n[ERROR] Checkpoint not found at '{_config['model_checkpoint']}'!")
        print("Please run training first (bash train.sh).\n")
        exit(1)

    classifier = IntentClassification(config_file)

    test_messages = [
        "I lost my credit card, what should I do?",
        "How do I top up my account?",
        "Why was I charged twice for the same transaction?",
        "Can you tell me my account balance?",
        "I want to cancel my subscription.",
    ]

    print("\n" + "=" * 60)
    print(" INFERENCE - BANKING INTENT CLASSIFICATION")
    print("=" * 60)

    for msg in test_messages:
        label = classifier(msg)
        print(f"\nInput   : {msg}")
        print(f"Intent  : {label}")
        print("-" * 60)

    print("\nInference completed.")
    print("To view full evaluation report, run: bash evaluate.sh\n")
