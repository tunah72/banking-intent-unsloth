# Fine-Tuning Intent Detection Model with Banking Dataset

Dự án áp dụng kỹ thuật fine-tuning cho bài toán phân loại ý định (intent classification) sử dụng tập dữ liệu `BANKING77` và thư viện `Unsloth`.

## Cấu trúc thư mục

Toàn bộ dự án được tổ chức theo chuẩn bao gồm các thành phần: `scripts`, `configs` và dữ liệu lấy mẫu `sample_data`.

## 1. Cài đặt môi trường

Chạy lệnh sau để cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```
*(Lưu ý: Quá trình cài đặt `unsloth` có thể yêu cầu phiên bản `torch`, `xformers` và `triton` cụ thể. Khuyến nghị chạy dự án này trên môi trường Google Colab để đạt hiệu suất tốt nhất và ít gặp lỗi tương thích thư viện).*

## 2. Tiền xử lý dữ liệu

Chạy lệnh sau để tải tập dữ liệu, thực hiện lấy mẫu (Stratified Sampling) và lưu ra thư mục `sample_data/`:
```bash
python scripts/preprocess_data.py
```

## 3. Huấn luyện mô hình (Training)

Chạy lệnh bash script sau để huấn luyện mô hình. Mô hình sẽ đọc cấu hình từ `configs/train.yaml`:
```bash
bash train.sh
```

## 4. Suy luận (Inference)

Chạy lệnh bash script sau để nạp checkpoint và đưa ra dự đoán ý định (Intent) từ một câu do người dùng nhập vào. Mô hình đọc cấu hình từ `configs/inference.yaml`:
```bash
bash inference.sh
```

## 5. Video Báo cáo (Demo)

* [Link Video Demo trên Google Drive](#) (Sẽ cập nhật sau khi quay)
