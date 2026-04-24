import os
import yaml
import json
import torch
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
            
        print("Loading label mapping...")
        with open(self.config['label_mapping_path'], 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            # JSON keys are stored as strings, cast to int
            self.id2label = {int(k): v for k, v in mapping.items()}
            
        print(f"Loading model and tokenizer from {self.config['model_checkpoint']}...")
        self.model, self.tokenizer = FastSequenceClassificationModel.from_pretrained(
            model_name = self.config['model_checkpoint'],
            max_seq_length = self.config['max_seq_length'],
            dtype = None,
            load_in_4bit = True,
        )
        
        # Enable Unsloth fast inference (2x speedup)
        FastSequenceClassificationModel.for_inference(self.model)
        
    def __call__(self, message):
        """
        Predict the intent label for a given input message.
        Args:
            message: A string containing the customer query.
        Returns:
            predicted_label: The predicted intent label name.
        """
        inputs = self.tokenizer(message, return_tensors="pt", truncation=True, max_length=self.config['max_seq_length'])
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Disable gradient computation for memory efficiency
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        predicted_class_id = logits.argmax(-1).item()
        
        return self.id2label[predicted_class_id]

if __name__ == '__main__':
    config_file = "configs/inference.yaml"
    
    # Read config to get checkpoint path
    with open(config_file, 'r', encoding='utf-8') as f:
        _config = yaml.safe_load(f)
    
    # Check if checkpoint exists
    if not os.path.exists(_config['model_checkpoint']):
        print(f"\n[ERROR] Checkpoint not found at '{_config['model_checkpoint']}'!")
        print("Please run training first (bash train.sh).\n")
        exit(1)
        
    # Initialize classifier
    classifier = IntentClassification(config_file)
    
    # Static test messages
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
