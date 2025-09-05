from ultralytics import YOLO

# 모델 경로 명확히 설정 (절대경로 사용)
yolo_model_path = "/Users/abitria/coding/python/yolov8s_playing_cards.pt"

# 모델 로딩
model = YOLO(yolo_model_path)

# 클래스 확인
print("✅ 로드된 클래스 수:", len(model.names))
print("✅ 클래스 이름 목록:", model.names)
