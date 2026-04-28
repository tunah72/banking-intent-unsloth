# Banking Intent Classification with Unsloth & QLoRA

Fine-tunes **Llama-3.1 8B** để phân loại ý định khách hàng ngân hàng thành 77 nhóm sử dụng bộ dữ liệu **BANKING77**, thông qua kỹ thuật **QLoRA (4-bit quantization + LoRA adapters)** và thư viện **Unsloth**.

## Video Demonstration

Link demo:
https://drive.google.com/file/d/1G1LSgbqrZogLUFtkk_9AZBJoNq7PMFT0/view?usp=sharing

## Đường dẫn Model trên Hugging Face
https://huggingface.co/tunah/banking-intent

## Kết quả Fine-tune (Gồm sample_data, checkpoints, results)
https://drive.google.com/drive/folders/1N74gttIResyRa9b4yiVsiY7RP8oLE4nl?usp=sharing

## Công nghệ sử dụng

| Thành phần | Chi tiết |
|---|---|
| Base Model | `unsloth/Llama-3.1-8B-bnb-4bit` (LLaMA 3.1, 8B params) |
| Quantization | 4-bit NF4 (BitsAndBytes) + double quantization |
| Fine-tuning | QLoRA via PEFT — chỉ train LoRA adapters + classification head |
| Dataset | `PolyAI/banking77` — 77 banking intent labels |
| Framework | HuggingFace Transformers + PEFT + Unsloth |
| Evaluation | Scikit-learn (Accuracy, Macro F1, Confusion Matrix) |

---

## Cấu trúc thư mục

```text
banking-intent-unsloth/
├── scripts/
│   ├── preprocess_data.py   # Tải BANKING77, Stratified Sampling, xuất CSV
│   ├── train.py             # Fine-tuning với QLoRA (BitsAndBytes + PEFT)
│   ├── inference.py         # Class IntentClassification (OOP)
│   ├── evaluate.py          # Đánh giá toàn diện trên tập test
│   └── upload_to_hf.py      # Upload model checkpoints lên Hugging Face Hub
├── configs/
│   ├── train.yaml           # Hyperparameters: LoRA, Trainer, sampling
│   ├── inference.yaml       # Cấu hình inference: checkpoint path, seq_length
│   └── evaluate.yaml        # Cấu hình evaluation: checkpoint, output_dir
├── sample_data/
│   ├── train.csv            # ~3826 mẫu (50/label × 77 labels)
│   ├── val.csv              # ~375 mẫu (5/label × 77 labels)
│   ├── test.csv             # 770 mẫu (10/label × 77 labels)
│   └── label_mapping.json   # Ánh xạ {id: intent_label}
├── notebooks/
│   ├── colab_pipeline.ipynb # Pipeline đầy đủ cho Google Colab
│   └── kaggle_pipeline.ipynb# Pipeline đầy đủ cho Kaggle
├── results/                 # Thư mục chứa kết quả đánh giá (metrics, biểu đồ)
├── train.sh                 # Bash wrapper: python scripts/train.py
├── inference.sh             # Bash wrapper: python scripts/inference.py
├── evaluate.sh              # Bash wrapper: python scripts/evaluate.py
├── requirements.txt
└── README.md
```

---

## Hyperparameters

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Base Model | `unsloth/Llama-3.1-8B-bnb-4bit` | Pre-quantized 4-bit NF4 |
| LoRA rank (r) | 16 | |
| LoRA alpha | 32 | Scaling = alpha/r = 2× |
| LoRA target modules | q/k/v/o/gate/up/down_proj | Tất cả attention + MLP |
| modules_to_save | `score` | Classification head trainable |
| Batch size | 2 per device | T4 VRAM limit |
| Gradient accumulation | 4 steps | Effective batch = 8 |
| Learning rate | 2e-4 | |
| LR scheduler | linear + warmup 5% | |
| Epochs | 5 | |
| Max sequence length | 128 | Banking messages ngắn |
| Optimizer | adamw_8bit | 8-bit optimizer (tiết kiệm VRAM) |
| Weight decay | 0.01 | L2 regularization |

---

## Cài đặt môi trường

Yêu cầu: GPU với CUDA (khuyến nghị Google Colab T4 hoặc Kaggle T4 x2).

```bash
# 1. Clone repository
git clone https://github.com/tunah72/banking-intent-unsloth.git
cd banking-intent-unsloth

# 2. Cài đặt dependencies
pip install -r requirements.txt
```

> **Lưu ý CUDA**: Unsloth yêu cầu CUDA tương thích với PyTorch đang cài. Trên Colab/Kaggle, môi trường đã có sẵn CUDA. Trên máy local, xem hướng dẫn tại https://docs.unsloth.ai/get-started/installing-+-updating

---

## Pipeline thực thi

### Bước 1 — Tiền xử lý dữ liệu

Tải BANKING77 từ HuggingFace, áp dụng Stratified Sampling và chuẩn hóa văn bản (strip + lowercase):

```bash
python scripts/preprocess_data.py
```

Output: `sample_data/train.csv`, `sample_data/val.csv`, `sample_data/test.csv`, `sample_data/label_mapping.json`

### Bước 2 — Fine-tuning (QLoRA)

```bash
bash train.sh
```

Checkpoint được lưu tại `checkpoints/final_best_model/`. Training tự động tiếp tục từ checkpoint nếu bị gián đoạn.

### Bước 3 — Inference Demo

```bash
bash inference.sh
```

### Bước 4 — Đánh giá toàn diện

```bash
bash evaluate.sh
```

Output tại `results/`:
- `test_predictions.csv` — Log dự đoán đầy đủ
- `classification_report.txt` — Precision, Recall, F1 per label
- `metrics.json` — Metrics JSON
- `confusion_matrix.png` — Confusion matrix heatmap

### Bước 5 — Upload model lên Hugging Face Hub

Đăng nhập vào Hugging Face qua CLI:
```bash
huggingface-cli login
```
Chạy script để upload thư mục `checkpoints/final_best_model` lên Hub:
```bash
python scripts/upload_to_hf.py
```
*(Lưu ý: Mở file `scripts/upload_to_hf.py` và sửa `repo_id = "your_username/banking-intent"` cho phù hợp với tài khoản của bạn trước khi chạy).*

---

## Sử dụng class IntentClassification

```python
from scripts.inference import IntentClassification

# model_path trỏ đến file YAML cấu hình (không phải checkpoint trực tiếp)
classifier = IntentClassification("configs/inference.yaml")

# Dự đoán intent
result = classifier("I lost my credit card, what should I do?")
print(result)  # → "lost_or_stolen_card"

result = classifier("What is my account balance?")
print(result)  # → "balance_not_updated_after_bank_transfer"
```

---

## Kết quả

- **Test Accuracy:** `91.04%`
- **Macro F1-Score:** `90.95%`
