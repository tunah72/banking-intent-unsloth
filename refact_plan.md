# Kế hoạch Refactor Code

Danh sách các vấn đề cần sửa trước khi submit. Đổi `[ ]` thành `[x]` khi hoàn thành.

---

## 1. Sửa `metric_for_best_model` → `eval_accuracy`
- **File:** `configs/train.yaml`
- **Vấn đề:** Đang dùng `eval_loss` để chọn best checkpoint. Với bài 77-class classification, eval_loss không tương quan tốt với accuracy — model có loss thấp nhất chưa chắc là model chính xác nhất.
- **Sửa:**
  ```yaml
  metric_for_best_model: "eval_accuracy"
  greater_is_better: true
  ```
- [x] Hoàn thành

---

## 2. Thêm text normalization vào `preprocess_data.py`
- **File:** `scripts/preprocess_data.py`
- **Vấn đề:** `requirements.md` yêu cầu "Chuẩn hóa văn bản (text normalization)" nhưng script hiện tại không có bước này.
- **Sửa:** Thêm sau khi load dataset, trước khi sampling:
  ```python
  df_train['text'] = df_train['text'].str.strip().str.lower()
  df_test['text'] = df_test['text'].str.strip().str.lower()
  ```
- [x] Hoàn thành

---

## 3. Sửa `.gitignore` để `sample_data/*.csv` được commit
- **File:** `.gitignore`
- **Vấn đề:** Toàn bộ `sample_data/` đang bị ignore. `requirements.md` yêu cầu `sample_data/train.csv` và `sample_data/test.csv` phải có trong repo.
- **Sửa:** Thay dòng `sample_data/` bằng:
  ```gitignore
  # Chỉ ignore các file lớn, giữ lại CSV và label mapping
  sample_data/*.pkl
  sample_data/*.bin
  ```
  Sau đó commit các file CSV và label_mapping.json.
- [x] Hoàn thành

---

## 4. Tối ưu hyperparameter LoRA trong `configs/train.yaml`
- **File:** `configs/train.yaml`
- **Vấn đề (3 điểm):**
  - `lora_alpha: 16` (nên = 2×r = 32 để scale factor đạt 2.0)
  - `lora_dropout: 0` (không có regularization, dễ overfit với 3850 mẫu)
  - Thiếu `warmup_ratio` (LR 2e-4 bắt đầu ngay từ đỉnh, dễ bất ổn)
- **Sửa:**
  ```yaml
  lora_alpha: 32
  lora_dropout: 0.05

  training_args:
    warmup_ratio: 0.05
  ```
- [x] Hoàn thành

---

## 5. Sửa mô tả sai trong `README.md`
- **File:** `README.md`
- **Vấn đề:** Step 3 (Inference) mô tả *"Launches a CLI chatbot that allows you to type custom banking queries"* — sai hoàn toàn. `inference.py` dùng danh sách câu cố định (static list), không có `input()`.
- **Sửa:** Đổi mô tả thành:
  > *"Runs inference on a predefined set of sample banking queries and prints the predicted intent for each."*
- [x] Hoàn thành

---

## 6. Thêm `seed` vào `configs/train.yaml`
- **File:** `configs/train.yaml`
- **Vấn đề:** LoRA có `random_state=3407` nhưng `TrainingArguments` không có `seed` → kết quả không tái lập được giữa các lần chạy.
- **Sửa:** Thêm vào `training_args`:
  ```yaml
  training_args:
    seed: 42
  ```
  Và trong `train.py`, truyền `seed=t_args['seed']` vào `TrainingArguments`.
- [x] Hoàn thành

---

## 7. Fix DeprecationWarning Pandas 2.x trong `preprocess_data.py`
- **File:** `scripts/preprocess_data.py`, dòng 61
- **Vấn đề:** `groupby().apply(lambda x: x.sample(...))` gây DeprecationWarning trên Pandas 2.2+ vì lambda dùng group column.
- **Sửa:**
  ```python
  sampled_test = df_test.groupby('label', group_keys=False).apply(
      lambda x: x.sample(n=min(len(x), n_test_samples), random_state=42),
      include_groups=False
  ).reset_index(drop=True)
  ```
  Lưu ý: `include_groups=False` chỉ available từ Pandas 2.2+. Nếu cần tương thích rộng hơn, dùng vòng lặp như phần train/val ở trên.
- [x] Hoàn thành
