import os
import json
import yaml
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoConfig
from unsloth import FastSequenceClassificationModel


class IntentClassification:
    def __init__(self, model_path):
        """
        Initialize the model from a YAML configuration file.
        Args:
            model_path: Path to the inference configuration file (e.g. configs/inference.yaml).
        """
        print(f"Loading configuration from {model_path}...")
        with open(model_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        checkpoint_dir = self.config['model_checkpoint']

        print("Loading label mapping...")
        with open(self.config['label_mapping_path'], 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        self.id2label = {int(k): v for k, v in mapping.items()}

        # Read base model name and num_labels from the saved checkpoint metadata
        print(f"Reading checkpoint metadata from {checkpoint_dir}...")
        peft_cfg        = PeftConfig.from_pretrained(checkpoint_dir)
        model_cfg       = AutoConfig.from_pretrained(checkpoint_dir)
        base_model_name = peft_cfg.base_model_name_or_path
        num_labels      = model_cfg.num_labels

        print(f"Loading base model with Unsloth: {base_model_name}...")
        base_model, self.tokenizer = FastSequenceClassificationModel.from_pretrained(
            model_name=base_model_name,
            num_labels=num_labels,
            max_seq_length=self.config['max_seq_length'],
            dtype=None,
            load_in_4bit=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

        print("Applying LoRA adapters...")
        self.model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        self.model.eval()

        # Enable Unsloth's native 2x faster inference — avoids OOM during prediction
        print("Enabling fast inference mode...")
        FastSequenceClassificationModel.for_inference(self.model)
        print("Model ready.")

    def __call__(self, message):
        """
        Predict the intent label for a given input message.
        Args:
            message: A string containing the customer query.
        Returns:
            predicted_label: The predicted intent label name.
        """
        inputs = self.tokenizer(
            message,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['max_seq_length'],
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

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
