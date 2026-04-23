import os
import yaml
import json
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from unsloth import FastSequenceClassificationModel

class IntentClassification:
    def __init__(self, model_path):
        """
        Khởi tạo mô hình dựa trên đường dẫn tới file cấu hình inference.yaml.
        Yêu cầu của Đề bài: Hàm này nhận model_path là đường dẫn tới cấu hình.
        """
        print(f"Loading configuration from {model_path}...")
        with open(model_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        print("Loading label mapping...")
        with open(self.config['label_mapping_path'], 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            # Key của JSON lưu dạng string, cần ép về int
            self.id2label = {int(k): v for k, v in mapping.items()}
            
        print(f"Loading model and tokenizer from {self.config['model_checkpoint']}...")
        self.model, self.tokenizer = FastSequenceClassificationModel.from_pretrained(
            model_name = self.config['model_checkpoint'],
            max_seq_length = self.config['max_seq_length'],
            dtype = None,
            load_in_4bit = True,
        )
        
        # Tối ưu hóa cho quá trình suy luận (Inference) tăng tốc độ x2
        FastSequenceClassificationModel.for_inference(self.model)
        
    def __call__(self, message):
        """
        Dự đoán nhãn intent cho một đoạn tin nhắn đầu vào.
        Yêu cầu của Đề bài: Nhận message và trả về predicted_label.
        """
        inputs = self.tokenizer(message, return_tensors="pt", truncation=True, max_length=self.config['max_seq_length'])
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Bỏ qua lưu trữ đạo hàm để tiết kiệm RAM và tăng tốc độ
        with torch.no_grad(): # Hoặc torch.inference_mode()
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        predicted_class_id = logits.argmax(-1).item()
        
        return self.id2label[predicted_class_id]

if __name__ == '__main__':
    config_file = "configs/inference.yaml"
    
    # Cảnh báo nếu bạn lỡ chạy file này trước khi Train
    if not os.path.exists("checkpoints/final_best_model"):
        print("\n[!] CẢNH BÁO: Chưa tìm thấy checkpoint mô hình đã huấn luyện!")
        print("Vui lòng chạy quá trình Train (bash train.sh) trên Colab trước khi chạy Inference.\n")
        exit(1)
        
    # Khởi tạo Class
    classifier = IntentClassification(config_file)
    
    # ---------------------------------------------------------
    # BƯỚC 1: DEMO TƯƠNG TÁC (GIAO LƯU NHẬP TỪ BÀN PHÍM)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" BƯỚC 1: GIAO LƯU TRỰC TIẾP (INTERACTIVE DEMO)")
    print("="*60)
    print("Hãy nhập một câu hỏi/phàn nàn bằng tiếng Anh.")
    print("(Gõ 'quit', 'exit' hoặc 'q' để chuyển sang Bước 2).")
    
    while True:
        user_input = input("\n[Khách hàng]: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if user_input.strip() == "":
            continue
            
        predicted_intent = classifier(user_input)
        print(f"[Hệ thống] Dự đoán Intent: >> {predicted_intent} <<")
        
    # ---------------------------------------------------------
    # BƯỚC 2: CHẤM ĐIỂM ACCURACY TRÊN TẬP TEST
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" BƯỚC 2: CHẤM ĐIỂM TRÊN TẬP TEST (FINAL ACCURACY)")
    print("="*60)
    
    test_path = classifier.config.get('test_data_path', 'sample_data/test.csv')
    if os.path.exists(test_path):
        print(f"Đang đọc dữ liệu kiểm thử từ {test_path}...")
        df_test = pd.read_csv(test_path)
        
        y_true_names = df_test['label_name'].tolist()
        y_pred_names = []
        
        print(f"Đang tiến hành dự đoán {len(df_test)} mẫu. Vui lòng chờ...")
        for idx, row in df_test.iterrows():
            pred_name = classifier(row['text'])
            y_pred_names.append(pred_name)
            
        accuracy = accuracy_score(y_true_names, y_pred_names)
        
        print("\n" + "*"*40)
        print(f" ĐỘ CHÍNH XÁC (TEST ACCURACY): {accuracy * 100:.2f}%")
        print("*"*40 + "\n")
    else:
        print(f"Không tìm thấy file {test_path} để đánh giá.")
