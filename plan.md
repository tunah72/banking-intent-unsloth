# Kế hoạch Thực hiện Dự án: Fine-Tuning Intent Detection

Dưới đây là kế hoạch chi tiết từng bước. Bạn có thể sử dụng file này như một danh sách công việc (todo list) để theo dõi tiến độ. Thay đổi `[ ]` thành `[x]` khi hoàn thành một phần.

## Giai đoạn 1: Chuẩn bị Môi trường & Khung sườn
- [x] Phân tích yêu cầu và chốt cấu trúc dự án.
- [x] Khởi tạo khung sườn (directories & empty files).
- [x] Soạn thảo file `requirements.txt` với các thư viện cần thiết (Unsloth, Transformers, v.v.).
- [x] Soạn thảo khung sườn ban đầu cho file `README.md`.

## Giai đoạn 2: Tiền xử lý dữ liệu (Data Preprocessing)
- [x] Viết script `scripts/preprocess_data.py`:
  - Tải bộ dữ liệu `BANKING77` từ HuggingFace.
  - Áp dụng Stratified Sampling theo chiến lược: **50 mẫu/nhãn cho Train**, **5 mẫu/nhãn cho Validation**, và **10 mẫu/nhãn cho Test**.
  - Map 77 nhãn văn bản thành các số ID (từ 0 đến 76) để phù hợp cho bài toán Sequence Classification.
- [ ] Chạy script để xuất ra kết quả `sample_data/train.csv`, `sample_data/val.csv` và `sample_data/test.csv`.

## Giai đoạn 3: Huấn luyện Mô hình (Training Model)
- [x] **Kịch bản Thử nghiệm (Experiment Plan) theo thứ tự ưu tiên:**
  1. **Llama-3 8B** (`unsloth/llama-3-8b-bnb-4bit`): Ưu tiên số 1, chạy ổn định, độ chính xác cao nhất trên T4.
  2. **Qwen 2.5 7B** (`unsloth/qwen2.5-7b-bnb-4bit`): Dự phòng thử nghiệm nếu có thời gian, đánh giá sức mạnh của kiến trúc mới nhất.
  3. **Gemma 2 9B** (`unsloth/gemma-2-9b-bnb-4bit`): Dự phòng chạy cuối cùng (cần lưu ý giảm `batch_size=1` vì mô hình ngốn nhiều VRAM nhất).
- [x] Viết cấu hình `configs/train.yaml`: Khai báo tên mô hình gốc, tham số batch_size, learning_rate, các tham số LoRA (r, alpha). Bổ sung thiết lập lưu checkpoint tự động và giữ lại model tốt nhất (`load_best_model_at_end=True`).
- [x] Viết script `scripts/train.py`:
  - Khởi tạo tokenizer và model với Unsloth theo dạng Sequence Classification.
  - Xây dựng Pipeline Dataset của HuggingFace từ file CSV (đã đổi tên cột mục tiêu thành `labels` theo chuẩn).
  - Cấu hình Trainer và chạy huấn luyện. (Đã tích hợp Resume Checkpoint và hàm `compute_metrics` để hiển thị Accuracy).
  - Lưu Checkpoint sau khi hoàn tất.
- [x] Viết nội dung chạy lệnh cho file `train.sh`.
- [ ] **Kết nối Colab MCP:** Đưa mã nguồn lên Colab, chạy lệnh `bash train.sh` trên GPU và tải Checkpoint về máy local.

## Giai đoạn 4: Suy luận (Inference Implementation)
- [x] Viết cấu hình `configs/inference.yaml` chứa: `model_checkpoint`, `max_seq_length`, `label_mapping_path` và `test_data_path`.
- [x] Viết script `scripts/inference.py`:
  - Khởi tạo class OOP `IntentClassification` đúng chuẩn với 2 hàm `__init__` (nhận tham số file config) và `__call__` (nhận tham số text).
  - **Lưu ý tối ưu:** Đã nạp hàm `FastSequenceClassificationModel.for_inference(model)` giúp tăng x2 tốc độ và đặt lệnh dự đoán bên trong `torch.no_grad()` để tránh lỗi tràn RAM (OOM).
  - **Kịch bản Demo (Dùng cho Video):** Viết khối lệnh test tự động chạy Bước 1 (Giao lưu tương tác) bằng lệnh `input()` từ Terminal.
- [x] Viết nội dung chạy lệnh cho file `inference.sh`.
- [x] **Xây dựng luồng Đánh giá Học thuật (Evaluation Phase)** để làm báo cáo:
  - Viết file `configs/evaluate.yaml` và `scripts/evaluate.py`.
  - Kết quả 1: Lưu toàn bộ lịch sử dự đoán của tập Test ra `results/test_predictions.csv` để đối chiếu và nghiên cứu lỗi sai.
  - Kết quả 2: Xuất `classification_report.txt` và `metrics.json` (chứa Precision, Recall, F1).
  - Kết quả 3: Vẽ biểu đồ `confusion_matrix.png` siêu đẹp bằng matplotlib/seaborn.
  - Viết file chạy lệnh `evaluate.sh`.
- [ ] Chạy thử nghiệm file suy luận ở máy local hoặc Colab để xác minh kết quả và ghi nhận độ chính xác.

## Giai đoạn 5: Hoàn thiện Báo cáo
- [x] Cập nhật `README.md` lần cuối (mô tả kiến trúc, luồng chạy 4 bước, thư viện và placeholder video).
- [ ] Quay Video Demo chứng minh việc tải model và phân loại câu, sau đó upload lên Google Drive.
- [ ] Gắn link Video vào `README.md`.
- [ ] Commit và push toàn bộ cấu trúc dự án lên GitHub.
