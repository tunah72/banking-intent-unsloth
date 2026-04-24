import os
import yaml
import json
import torch
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
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        predicted_class_id = logits.argmax(-1).item()
        
        return self.id2label[predicted_class_id]

if __name__ == '__main__':
    config_file = "configs/inference.yaml"
    
    # Đọc config trước để lấy đường dẫn checkpoint
    with open(config_file, 'r', encoding='utf-8') as f:
        _config = yaml.safe_load(f)
    
    # Cảnh báo nếu chưa có checkpoint
    if not os.path.exists(_config['model_checkpoint']):
        print(f"\n[!] CẢNH BÁO: Chưa tìm thấy checkpoint tại '{_config['model_checkpoint']}'!")
        print("Vui lòng chạy quá trình Train (bash train.sh) trên Colab trước khi chạy Inference.\n")
        exit(1)
        
    # Khởi tạo Class
    classifier = IntentClassification(config_file)
    
    # ---------------------------------------------------------
    # DEMO TƯƠNG TÁC (GIAO LƯU NHẬP TỪ BÀN PHÍM)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" INTERACTIVE DEMO - BANKING INTENT CLASSIFICATION")
    print("="*60)
    print("Hãy nhập một câu hỏi/phàn nàn bằng tiếng Anh.")
    print("(Gõ 'quit', 'exit' hoặc 'q' để thoát).\n")
    
    while True:
        user_input = input("[Khách hàng]: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if user_input.strip() == "":
            continue
            
        predicted_intent = classifier(user_input)
        print(f"[Hệ thống] Dự đoán Intent: >> {predicted_intent} <<\n")
        
    print("[Hệ thống] Đã thoát chế độ Giao lưu.")
    print("Để xem báo cáo độ chính xác trên toàn bộ tập Test, vui lòng chạy: bash evaluate.sh\n")
