# Phân tích Yêu cầu Dự án 2: Fine-Tuning Intent Detection Model

Dựa trên tài liệu mô tả đồ án, dưới đây là chi tiết các yêu cầu cần thực hiện và những lưu ý quan trọng để hoàn thành tốt đồ án NLP này.

## 1. Yêu cầu chi tiết (Task Requirements)

### 1.1. Chuẩn bị và xử lý dữ liệu (Data Preparation and Processing)
- **Tập dữ liệu:** Sử dụng bộ dữ liệu **BANKING77**.
- **Lấy mẫu dữ liệu (Sampling):** Không sử dụng toàn bộ dữ liệu. Cần lấy một tập con (subset) đủ nhỏ để có thể huấn luyện mô hình vừa vặn với tài nguyên máy tính hiện có (ví dụ: Google Colab T4 GPU).
- **Tiền xử lý:**
  - Chuẩn hóa văn bản (text normalization).
  - Ánh xạ nhãn (label mapping).
  - Làm sạch dữ liệu cơ bản.
- **Định dạng dữ liệu:** Chuyển đổi các nhãn ý định (intent labels) sang dạng phù hợp với bài toán Phân loại chuỗi (Sequence Classification).
- **Chia tập dữ liệu:** Phân chia tập dữ liệu mẫu thành 2 tập `train` và `test`. (Khuyến khích tạo thêm tập `validation` từ tập `train` để theo dõi quá trình đánh giá và chọn lọc mô hình).

### 1.2. Huấn luyện mô hình (Fine-tuning with Unsloth)
- **Công cụ:** Sử dụng thư viện **Unsloth** để fine-tune mô hình. Bám sát hướng dẫn từ trang chủ của Unsloth.
- **Môi trường:** Google Colab, Kaggle hoặc máy local.
- **Báo cáo Siêu tham số (Hyperparameters):** Phải lưu trữ và mô tả rõ ràng các thông số sau:
  - Batch size.
  - Learning rate.
  - Optimizer.
  - Số bước huấn luyện (training steps) hoặc epochs.
  - Độ dài chuỗi tối đa (maximum sequence length).
  - Bất kỳ kỹ thuật hiệu chỉnh (regularization) hay tăng cường dữ liệu (augmentation) nào được sử dụng.
- **Lưu trữ:** Phải lưu lại Model Checkpoint sau khi hoàn thành quá trình fine-tune.

### 1.3. Xây dựng module Suy luận (Inference Implementation)
- **File thực thi:** Viết một file inference độc lập có chức năng tải checkpoint và dự đoán nhãn cho câu đầu vào.
- **Cấu trúc OOP bắt buộc:** Phải xây dựng class `IntentClassification` với chính xác 2 hàm:
  - `__init__(self, model_path)`: Đọc file cấu hình (config file), khởi tạo tokenizer và nạp checkpoint của mô hình.
  - `__call__(self, message)`: Nhận một chuỗi văn bản (message) và trả về nhãn ý định được dự đoán (predicted label).
- **Lưu ý đối số `model_path`:** Biến này phải trỏ đến một file cấu hình (ví dụ `.yaml`), và trong file cấu hình đó mới lưu đường dẫn thực sự trỏ tới checkpoint.
- **Ví dụ gọi hàm:** Viết kèm một đoạn code nhỏ để minh họa việc khởi tạo class và gọi dự đoán.

### 1.4. Mã nguồn và Tổ chức dự án (Source Code)
- Toàn bộ mã nguồn phải được đẩy lên **GitHub**.
- **Cấu trúc thư mục bắt buộc:**
```text
banking-intent-unsloth
    |-- scripts
    |        |-- train.py
    |        |-- inference.py
    |        |-- preprocess_data.py
    |
    |-- configs
    |        |-- train.yaml
    |        |-- inference.yaml
    |
    |-- sample_data
    |        |-- train.csv
    |        |-- test.csv
    |
    |-- train.sh
    |-- inference.sh
    |-- requirements.txt
    |-- README.md
```
- **File README.md:** Phải cung cấp đầy đủ hướng dẫn để: cài đặt môi trường, tải dữ liệu, chạy huấn luyện (train), và chạy suy luận (inference).

### 1.5. Video Báo cáo (Video Demonstration)
- Cần quay một đoạn video ngắn từ **2 - 5 phút** (không cần biên tập/chỉnh sửa phức tạp).
- **Nội dung bắt buộc trong video:**
  - Demo cách thực thi file/script inference.
  - Demo ít nhất 1 câu đầu vào và xuất ra màn hình kết quả dự đoán ý định (predicted intent label).
  - In ra/hiển thị độ chính xác cuối cùng (Final Accuracy) của mô hình trên tập test.
- **Nộp bài:** Upload video lên **Google Drive**, cấp quyền xem công khai (Public) và gắn link này vào file `README.md`.

---

## 2. Phân tích & Các Lưu ý Quan trọng khi Thực hiện (Execution Notes)

1. **Quản lý tài nguyên phần cứng (OOM - Out of Memory):**
   - Các mô hình LLM rất tốn VRAM. Việc bắt buộc phải lấy mẫu dữ liệu (sample data) là để tránh OOM và giảm thời gian train. Cần dùng chiến lược `Stratified Sampling` để lấy tập con mà vẫn giữ được sự cân bằng về tỷ lệ giữa các nhãn (77 nhãn).
   - Tinh chỉnh tham số `max_sequence_length` một cách cẩn thận. Do dữ liệu banking chủ yếu là các câu hỏi/tin nhắn ngắn, có thể giới hạn seq_length ở mức 64, 128 hoặc 256 để tối ưu tốc độ và VRAM.

2. **Sử dụng Unsloth:**
   - Unsloth tập trung vào việc tối ưu LLM fine-tuning qua kỹ thuật (Q)LoRA. Đảm bảo cấu hình đúng các tham số LoRA (r, lora_alpha, target_modules) sao cho mô hình chỉ cập nhật các adapter thay vì toàn bộ tham số.
   - Khi chuẩn bị dữ liệu cho LLM phân loại (Text Classification), cần chuyển đổi nhãn thành cấu trúc câu prompt dạng hướng dẫn (Instruction-based) hoặc sử dụng các class dành riêng cho Sequence Classification được hỗ trợ bởi transformers/unsloth.

3. **Cấu trúc tham số cho Class Inference:**
   - Đề bài chỉ định hàm khởi tạo `__init__(self, model_path)`. Hãy lưu ý `model_path` ở đây được đề bài gợi ý là *đường dẫn đến tệp cấu hình*. Nghĩa là ta sẽ truyền vào `configs/inference.yaml`. 
   - Mã giả xử lý trong hàm `__init__`:
     ```python
     import yaml
     class IntentClassification:
         def __init__(self, model_path):
             with open(model_path, 'r') as file:
                 config = yaml.safe_load(file)
             checkpoint_dir = config['model_checkpoint']
             # Thực hiện load tokenizer và model từ checkpoint_dir
             ...
     ```

4. **Tổ chức Scripts & Automation:**
   - Đề tài yêu cầu file `.sh` (`train.sh`, `inference.sh`). Đây là các bash script để bọc (wrap) việc gọi mã python chạy tự động. 
   - Ví dụ: `train.sh` sẽ chứa dòng lệnh `python scripts/train.py --config configs/train.yaml`. Điều này giúp việc vận hành (pipeline) chuyên nghiệp hơn giống với tiêu chuẩn trong công nghiệp (NLP Industry).

5. **Lưu ý file README.md:**
   - Bắt buộc phải có hướng dẫn cài đặt thư viện (`pip install -r requirements.txt`). Nên ghi chú rõ phiên bản CUDA cần thiết để cài đặt thư viện Unsloth nhằm tránh xung đột thư viện xformers/flash-attn.
   - Tuyệt đối không quên chèn link video Google Drive vào README. Đừng để link private, giám khảo không xem được sẽ mất điểm phần video.
