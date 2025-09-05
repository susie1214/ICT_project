import os
import queue
import sounddevice as sd
import numpy as np
from google.cloud import speech
from ultralytics import YOLO
import cv2, cvzone, serial
import time
# -------- [1] 환경 변수 (API 키 설정) --------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/abitria/coding/python/stt.json"
# -------- [2] 카드 클래스명 (YOLO 학습 순서) --------
classNames = [
    '10C','10D','10H','10S','2C','2D','2H','2S',
    '3C','3D','3H','3S','4C','4D','4H','4S',
    '5C','5D','5H','5S','6C','6D','6H','6S',
    '7C','7D','7H','7S','8C','8D','8H','8S',
    '9C','9D','9H','9S','AC','AD','AH','AS',
    'JC','JD','JH','JS','KC','KD','KH','KS',
    'QC','QD','QH','QS'
]
# -------- [3] 카드명 매핑 (음성 → 카드코드) --------
card_map = {
    "spade ace": "AS", "spade two": "2S", "spade three": "3S", "spade four": "4S", "spade five": "5S",
    "spade six": "6S", "spade seven": "7S", "spade eight": "8S", "spade nine": "9S", "spade ten": "10S",
    "spade jack": "JS", "spade j": "JS", "spade queen": "QS", "spade king": "KS",
    "heart ace": "AH", "heart two": "2H", "heart three": "3H", "heart four": "4H", "heart five": "5H",
    "heart six": "6H", "heart seven": "7H", "heart eight": "8H", "heart nine": "9H", "heart ten": "10H",
    "heart jack": "JH", "heart j": "JH", "heart queen": "QH", "heart king": "KH",
    "diamond ace": "AD", "diamond two": "2D", "diamond three": "3D", "diamond four": "4D", "diamond five": "5D",
    "diamond six": "6D", "diamond seven": "7D", "diamond eight": "8D", "diamond nine": "9D", "diamond ten": "10D",
    "diamond jack": "JD", "diamond j": "JD", "diamond queen": "QD", "diamond king": "KD",
    "club ace": "AC", "club two": "2C", "club three": "3C", "club four": "4C", "club five": "5C",
    "club six": "6C", "club seven": "7C", "club eight": "8C", "club nine": "9C", "club ten": "10C",
    "club jack": "JC", "club j": "JC", "club queen": "QC", "club king": "KC"
}
# -------- [4] 실시간 마이크 입력 → 카드코드 추출 --------
def get_card_code_from_mic():
    RATE = 16000
    CHUNK = int(RATE / 10)
    audio_q = queue.Queue()
    def callback(indata, frames, time_info, status):
        audio_q.put(bytes(indata))
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US"
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=False
    )
    with sd.RawInputStream(samplerate=RATE, blocksize=CHUNK, dtype='int16',
                           channels=1, callback=callback):
        print("말해주세요 (예: 'spade seven')")
        audio_generator = (audio_q.get() for _ in range(50))  # 약 5초
        requests = (speech.StreamingRecognizeRequest(audio_content=chunk)
                    for chunk in audio_generator)
        responses = client.streaming_recognize(streaming_config, requests)
        for response in responses:
            for result in response.results:
                if result.is_final:
                    transcript = result.alternatives[0].transcript.lower()
                    print("인식된 음성:", transcript)
                    for key, code in card_map.items():
                        if key in transcript:
                            print(f"인식된 카드 코드: {code}")
                            return code
                    print("카드 코드 매칭 실패")
                    return None
# -------- [5] YOLO로 카드 인식 + 시리얼 전송 (PASS 카드 제외) --------
def yolo_with_pass(skip_card, yolo_model_path, serial_port="/dev/tty.usbserial-0001", baud=115200):
    model = YOLO(yolo_model_path)
    ser = serial.Serial(serial_port, baud, timeout=1)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"PASS 카드: {skip_card}")
    sent_cards = set()
    frame_count = 0
    inference_interval = 5
    while True:
        success, img = cap.read()
        if not success:
            print("카메라 오류")
            break
        frame_count += 1
        detections = []
        if frame_count % inference_interval == 0:
            results = model(img, stream=True)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = classNames[cls]
                    if conf > 0.5:
                        detections.append((x1, y1, x2, y2, label, conf))
                        if label not in sent_cards:
                            if label == skip_card:
                                print(f"[PASS] {label} → 'P'")
                                ser.write(b'P')
                            else:
                                symbol = label[-1].upper()  # 마지막 글자: C/H/D/S
                                print(f"[SEND] {symbol} ({label})")
                                ser.write(symbol.encode())
                            sent_cards.add(label)
                            time.sleep(0.7)
        # 시각화
        for x1, y1, x2, y2, label, conf in detections:
            color = (0, 0, 255) if label == skip_card else (0, 255, 0)
            cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, t=2, colorC=color)
            cvzone.putTextRect(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                               scale=1, thickness=2, colorT=color)
        if skip_card:
            cv2.putText(img, f"PASS: {skip_card}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        cv2.imshow("YOLOv8 Card Detection", img)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
# ================== 실행부 ==================
if __name__ == "__main__":
    yolo_model_path = "/Users/abitria/coding/python/yolov8s_playing_cards.pt"
    skip_card = get_card_code_from_mic()
    if skip_card:
        yolo_with_pass(skip_card, yolo_model_path)
    else:
        print("음성으로부터 카드 코드 추출 실패")