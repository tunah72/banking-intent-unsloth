# Banking Intent Classification with Unsloth & QLoRA

This repository contains the source code for an industry-grade NLP pipeline that fine-tunes a Large Language Model (LLM) to classify customer support messages into 77 distinct banking intents using the **BANKING77** dataset.

## 🎯 Project Overview
This project demonstrates how to perform **Sequence Classification** using **Unsloth** and **HuggingFace Transformers**. We leverage Parameter-Efficient Fine-Tuning (PEFT) via LoRA to achieve state-of-the-art accuracy while training entirely on a free Google Colab T4 GPU (16GB VRAM).

### Core Technologies:
- **Base Model:** Llama-3 8B (`unsloth/llama-3-8b-bnb-4bit`)
- **Optimization:** QLoRA (4-bit quantization) & Unsloth `FastSequenceClassificationModel`
- **Dataset:** PolyAI/banking77
- **Evaluation:** Scikit-Learn (Accuracy, Macro F1, Confusion Matrix)

## 🎥 Video Demonstration
> **[INSERT GOOGLE DRIVE VIDEO LINK HERE]**

*(The video demonstrates loading the trained model, running interactive inference, and calculating the final test set accuracy).*

## 📁 Repository Structure
```text
├── configs/              # YAML Configuration files
│   ├── train.yaml        # LoRA & Trainer hyperparameters
│   ├── inference.yaml    # Inference settings
│   └── evaluate.yaml     # Evaluation settings
├── scripts/              # Python source code
│   ├── preprocess_data.py # Stratified sampling pipeline
│   ├── train.py          # HuggingFace Trainer script
│   ├── inference.py      # Interactive CLI inference
│   └── evaluate.py       # Classification report & metrics
├── requirements.txt      # Project dependencies
├── .gitignore            # Git exclusion rules
├── train.sh              # Bash script to run training
├── inference.sh          # Bash script to run inference
└── evaluate.sh           # Bash script to run evaluation
```

## 🛠️ Installation
It is highly recommended to run this project on a machine with a CUDA-enabled GPU (e.g., Google Colab).
```bash
# Clone the repository
git clone https://github.com/tunah72/banking-intent-unsloth.git
cd banking-intent-unsloth

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Execution Pipeline

### Step 1: Data Preprocessing
Downloads the BANKING77 dataset and applies Stratified Sampling (50 Train / 5 Validation / 10 Test per label) to output clean CSV files.
```bash
python scripts/preprocess_data.py
```

### Step 2: Model Fine-tuning (QLoRA)
Runs the HuggingFace Trainer using the `configs/train.yaml` configuration. Automatically saves the best checkpoint based on Validation Loss.
```bash
bash train.sh
```

### Step 3: Interactive Inference
Launches a CLI chatbot that allows you to type custom banking queries and instantly receive the predicted intent.
```bash
bash inference.sh
```

### Step 4: Academic Evaluation
Evaluates the fine-tuned model against the 770 test samples. Calculates Accuracy, Precision, Recall, Macro F1-Score, and generates a Confusion Matrix.
```bash
bash evaluate.sh
```
*Outputs will be saved in the `results/` directory.*

## 📊 Expected Results (Placeholder)
*(After training on Colab, update this section with your actual metrics)*
- **Validation Loss:** `[Value]`
- **Test Accuracy:** `[Value]%`
- **Macro F1-Score:** `[Value]`
