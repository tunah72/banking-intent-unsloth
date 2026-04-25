# Plan v2: Fine-Tuning Banking Intent Detection - Phân tích & Kế hoạch Đầy đủ

## 1. Phân tích Hiện trạng

### 1.1. Files đã có

| File | Trạng thái | Ghi chú |
|---|---|---|
| `scripts/preprocess_data.py` | ✅ Hoàn chỉnh | Stratified sampling 50/5/10 per label, đúng yêu cầu |
| `scripts/train.py` | ❌ Bug nghiêm trọng | Score head bị đóng băng → model không học được |
| `scripts/inference.py` | ⚠️ OOM tiềm ẩn | Load base model không có quantization → ~16GB VRAM |
| `scripts/evaluate.py` | ⚠️ OOM tiềm ẩn | Cùng vấn đề với inference.py |
| `configs/train.yaml` | ❌ Thiếu | Thiếu `modules_to_save` cho score head |
| `configs/inference.yaml` | ✅ Hoàn chỉnh | - |
| `configs/evaluate.yaml` | ✅ Hoàn chỉnh | - |
| `train.sh` | ✅ Hoàn chỉnh | - |
| `inference.sh` | ✅ Hoàn chỉnh | - |
| `evaluate.sh` | ✅ Hoàn chỉnh | - |
| `requirements.txt` | ⚠️ Cần cải thiện | Thiếu version pinning |
| `README.md` | ⚠️ Chưa xong | Thiếu link video Google Drive |
| `sample_data/*.csv` | ❌ Chưa có | Cần chạy preprocess_data.py |
| `test_bnb.py` | 🗑️ Xóa | File thử nghiệm tạm, không cần thiết |

### 1.2. Bugs Nghiêm Trọng Cần Sửa Ngay

#### BUG 1 (CRITICAL): Score Head Bị Đóng Băng – Model Không Học Được

**Vấn đề**: Trong `scripts/train.py`, flow hiện tại là:
1. Load model với 4-bit quantization
2. `prepare_model_for_kbit_training()` → **đóng băng TẤT CẢ parameters** (requires_grad = False)
3. `get_peft_model(model, lora_config)` → chỉ mở LoRA adapter weights

Kết quả: Layer `model.score` (classification head, 77 classes) **hoàn toàn bị đóng băng** với weights ngẫu nhiên. LoRA adapters trong attention/MLP được train, nhưng đầu ra cuối cùng đi qua một linear layer frozen ngẫu nhiên → **model không thể học cách phân loại đúng**.

**Nguyên nhân gốc**: Thiếu `modules_to_save=["score"]` trong `LoraConfig`. Đây là tham số tiêu chuẩn của PEFT để đánh dấu classification head là trainable và lưu trọng số của nó cùng adapter.

**Fix**:
```python
# configs/train.yaml
modules_to_save:
  - "score"

# scripts/train.py - trong LoraConfig
lora_config = LoraConfig(
    ...
    modules_to_save=config.get('modules_to_save', ["score"]),
)
```

#### BUG 2 (MINOR): Tham số `llm_int8_skip_modules` Sai Context

**Vấn đề**: `llm_int8_skip_modules` là tham số của **int8** quantization (`load_in_8bit=True`), không áp dụng cho 4-bit NF4 (`load_in_4bit=True`). Trong mode 4-bit, tham số này bị ignore hoặc gây deprecation warning.

**Fix**: Xóa `llm_int8_skip_modules=["score"]` khỏi `BitsAndBytesConfig`.

#### BUG 3 (OOM): Inference/Evaluate Load Model Không Có Quantization

**Vấn đề**: `inference.py` và `evaluate.py` load base model:
```python
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,  # "unsloth/Llama-3.1-8B-bnb-4bit"
    torch_dtype=compute_dtype,  # float16, không có quantization
)
```
8B parameters × 2 bytes = **~16GB VRAM** chỉ riêng base model, vượt giới hạn T4 (16GB) khi tính thêm runtime overhead.

**Fix**: Thêm `BitsAndBytesConfig(load_in_4bit=True)` vào inference.py và evaluate.py khi load base model.

---

## 2. Kế hoạch Chi Tiết (10 Tasks)

---

### Task 1: Fix `configs/train.yaml` – Thêm `modules_to_save`

**File**: `configs/train.yaml`  
**Thay đổi**: Thêm key `modules_to_save: ["score"]` để classification head được train và lưu cùng adapter.

```yaml
# Thêm sau target_modules
modules_to_save:
  - "score"
```

---

### Task 2: Fix `scripts/train.py` – Sửa 2 Bugs

**File**: `scripts/train.py`

**Thay đổi 1**: Xóa `llm_int8_skip_modules` khỏi `BitsAndBytesConfig`:
```python
# TRƯỚC (sai):
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["score"],  # XÓA DÒNG NÀY
)

# SAU (đúng):
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)
```

**Thay đổi 2**: Thêm `modules_to_save` vào `LoraConfig`:
```python
# TRƯỚC (thiếu):
lora_config = LoraConfig(
    r=config['lora_r'],
    lora_alpha=config['lora_alpha'],
    target_modules=config['target_modules'],
    lora_dropout=config['lora_dropout'],
    bias="none",
    task_type=TaskType.SEQ_CLS,
)

# SAU (đúng):
lora_config = LoraConfig(
    r=config['lora_r'],
    lora_alpha=config['lora_alpha'],
    target_modules=config['target_modules'],
    lora_dropout=config['lora_dropout'],
    bias="none",
    task_type=TaskType.SEQ_CLS,
    modules_to_save=config.get('modules_to_save', ["score"]),  # THÊM DÒNG NÀY
)
```

---

### Task 3: Fix `scripts/inference.py` – Thêm Quantization Khi Load Base Model

**File**: `scripts/inference.py`

**Thay đổi**: Thêm `BitsAndBytesConfig` khi gọi `AutoModelForSequenceClassification.from_pretrained`:
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    num_labels=num_labels,
    device_map="auto",
    torch_dtype=compute_dtype,
    quantization_config=quantization_config,  # THÊM DÒNG NÀY
)
```

Khi `modules_to_save=["score"]` được dùng khi training, `PeftModel.from_pretrained` sẽ tự động restore score head weights từ adapter. Base model chỉ cần đúng architecture, không cần score weights chính xác.

---

### Task 4: Fix `scripts/evaluate.py` – Cùng Fix Với Inference

**File**: `scripts/evaluate.py`  
**Thay đổi**: Giống Task 3 – thêm `BitsAndBytesConfig` khi load base model.

---

### Task 5: Update `requirements.txt` – Pin Versions

**File**: `requirements.txt`  
**Thay đổi**: Thêm version constraints để tránh compatibility issues:

```
# Core ML
torch>=2.1.0
transformers>=4.46.0,<5.0.0
datasets==2.21.0
peft>=0.12.0
bitsandbytes>=0.43.0
accelerate>=0.26.0

# Unsloth (cần CUDA, cài theo hướng dẫn README)
unsloth

# Data processing
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0

# Config & Utils
pyyaml>=6.0
tqdm>=4.65.0
sentencepiece
protobuf

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

### Task 6: Generate `sample_data/` Files

**Lệnh**: `python scripts/preprocess_data.py`

**Output**:
- `sample_data/train.csv` – 3850 rows (50 × 77 labels)
- `sample_data/val.csv` – 385 rows (5 × 77 labels)
- `sample_data/test.csv` – 770 rows (10 × 77 labels)
- `sample_data/label_mapping.json` – ánh xạ `{id: label_name}` cho 77 intents

**Lưu ý**: BANKING77 chỉ có ~10k train samples. Với 50+5=55 samples/label, một số label có thể thiếu → script xử lý bằng `frac=1` (lấy tất cả nếu không đủ).

---

### Task 7: Update `.gitignore` – Loại `test_bnb.py` và Thêm Rules

**File**: `.gitignore`  
**Thay đổi**:
- Thêm `test_bnb.py` vào ignore (hoặc xóa file)
- Đảm bảo `sample_data/*.csv` và `sample_data/*.json` được commit (không ignore)
- Giữ nguyên `checkpoints/` trong ignore (model weights quá nặng)

---

### Task 8: Update Notebooks – Đồng bộ với Fixed Scripts

**Files**: `notebooks/colab_pipeline.ipynb`, `notebooks/kaggle_pipeline.ipynb`

Đảm bảo notebooks phản ánh đúng flow sau khi sửa bugs:
1. Cell install: `!pip install -r requirements.txt`
2. Cell preprocess: `!python scripts/preprocess_data.py`
3. Cell train: `!bash train.sh`
4. Cell evaluate: `!bash evaluate.sh`
5. Cell inference: `!bash inference.sh`

Notebooks cần được cập nhật để có thể chạy end-to-end trên Colab T4 hoặc Kaggle T4 x2.

---

### Task 9: Update `README.md` – Hoàn thiện Documentation

**File**: `README.md`  
**Thay đổi**:
- Cập nhật model name chính xác: `unsloth/Llama-3.1-8B-bnb-4bit`
- Thêm note về CUDA version requirement cho unsloth
- Giữ placeholder video link rõ ràng: `> **[VIDEO GOOGLE DRIVE - CẦN CẬP NHẬT SAU KHI QUAY]**`
- Thêm bảng hyperparameters theo yêu cầu

---

### Task 10: Thực thi Trên Cloud & Hoàn thiện Cuối

**Thứ tự thực hiện trên Colab/Kaggle**:
1. Upload/clone repo
2. `pip install -r requirements.txt`
3. `python scripts/preprocess_data.py`
4. `bash train.sh` (training ~30-60 phút trên T4)
5. `bash evaluate.sh` (ghi lại Final Accuracy)
6. `bash inference.sh` (demo inference)
7. Quay video màn hình (2-5 phút)
8. Upload video lên Google Drive, set public
9. Cập nhật link vào `README.md`
10. `git add . && git commit && git push`

---

## 3. Thứ Tự Ưu Tiên Thực Hiện

```
[NGAY LẬP TỨC - Sửa bugs trước khi training]
  Task 1 → Task 2 → Task 3 → Task 4 → Task 5

[SAU ĐÓ - Chuẩn bị data & clean up]  
  Task 6 → Task 7

[SONG SONG - Cập nhật docs & notebooks]
  Task 8 → Task 9

[CUỐI CÙNG - Cloud execution]
  Task 10
```

---

## 4. Siêu tham số Báo cáo (Hyperparameters Summary)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Base Model | `unsloth/Llama-3.1-8B-bnb-4bit` | LLaMA 3.1, 8B params, pre-quantized 4-bit |
| Quantization | NF4 (4-bit) + double quant | Giảm VRAM từ ~16GB xuống ~5GB |
| LoRA rank (r) | 16 | Số chiều adapter |
| LoRA alpha | 32 | Scaling factor (alpha/r = 2x) |
| LoRA target | q/k/v/o/gate/up/down proj | Tất cả attention + MLP layers |
| LoRA dropout | 0 | Không dropout (data nhỏ) |
| modules_to_save | `["score"]` | Score head trainable + saved |
| Batch size | 2 (per device) | T4 VRAM limit |
| Grad accum steps | 4 | Effective batch = 8 |
| Learning rate | 2e-4 | Với adamw_8bit optimizer |
| LR scheduler | linear + warmup 5% | |
| Epochs | 5 | |
| Max seq length | 128 | Banking messages ngắn |
| Optimizer | adamw_8bit | bitsandbytes 8-bit optimizer |
| Weight decay | 0.01 | L2 regularization nhẹ |
| Eval strategy | every 50 steps | |
| Data: Train | 3850 rows | 50/label × 77 labels |
| Data: Val | 385 rows | 5/label × 77 labels |
| Data: Test | 770 rows | 10/label × 77 labels |

---

## 5. Luồng Dữ liệu Qua Model

```
Input text (string)
    ↓ tokenizer (max_length=128, truncation)
input_ids, attention_mask
    ↓ LlamaModel (frozen base, 4-bit NF4)
hidden_states [batch, seq_len, 4096]
    ↓ pooling (last non-padding token)
pooled [batch, 4096]
    ↓ score head: Linear(4096, 77) ← TRAINABLE via modules_to_save
    ↓ + LoRA adapters in attention/MLP layers ← TRAINABLE
logits [batch, 77]
    ↓ argmax
predicted_class_id (0-76)
    ↓ id2label mapping
predicted_intent_label (string)
```

---

## 6. Checklist Nộp Bài

- [ ] `scripts/train.py` – Bug cố định (modules_to_save + no llm_int8_skip_modules)
- [ ] `scripts/inference.py` – Thêm quantization khi load
- [ ] `scripts/evaluate.py` – Thêm quantization khi load
- [ ] `configs/train.yaml` – Có `modules_to_save`
- [ ] `sample_data/train.csv` – Generated
- [ ] `sample_data/test.csv` – Generated
- [ ] `checkpoints/final_best_model/` – Trained (không push lên GitHub)
- [ ] `README.md` – Có link video Google Drive thực
- [ ] Video 2-5 phút trên Google Drive (public)
- [ ] GitHub repo – Push toàn bộ code
