import os
import queue
import sounddevice as sd
import numpy as np
from google.cloud import speech
from ultralytics import YOLO
import cv2, cvzone, serial
import time
# -------- [1] 환경 변수 (GCP 인증 키 설정) --------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/abitria/coding/python/stt.json"
# -------- [2] YOLO 클래스명 --------
classNames = [
    '10C','10D','10H','10S','2C','2D','2H','2S',
    '3C','3D','3H','3S','4C','4D','4H','4S',
    '5C','5D','5H','5S','6C','6D','6H','6S',
    '7C','7D','7H','7S','8C','8D','8H','8S',
    '9C','9D','9H','9S','AC','AD','AH','AS',
    'JC','JD','JH','JS','KC','KD','KH','KS',
    'QC','QD','QH','QS'
]
# -------- [3] 한글 음성 → 카드 코드 매핑 --------
card_map = {
    "스페이드 에이스": "AS", "스페이드 투": "2S", "스페이드 쓰리": "3S", "스페이드 포": "4S", "스페이드 파이브": "5S",
    "스페이드 식스": "6S", "스페이드 세븐": "7S", "스페이드 에잇": "8S", "스페이드 나인": "9S", "스페이드 텐": "10S",
    "스페이드 킹": "KS", "스페이드 퀸": "QS", "스페이드 잭": "JS",
    "하트 에이스": "AH", "하트 투": "2H", "하트 쓰리": "3H", "하트 포": "4H", "하트 파이브": "5H",
    "하트 식스": "6H", "하트 세븐": "7H", "하트 에잇": "8H", "하트 나인": "9H", "하트 텐": "10H",
    "하트 킹": "KH", "하트 퀸": "QH", "하트 잭": "JH",
    "다이아 에이스": "AD", "다이아 투": "2D", "다이아 쓰리": "3D", "다이아 포": "4D", "다이아 파이브": "5D",
    "다이아 식스": "6D", "다이아 세븐": "7D", "다이아 에잇": "8D", "다이아 나인": "9D", "다이아 텐": "10D",
    "다이아 킹": "KD", "다이아 퀸": "QD", "다이아 잭": "JD",
    "클로버 에이스": "AC", "클로버 투": "2C", "클로버 쓰리": "3C", "클로버 포": "4C", "클로버 파이브": "5C",
    "클로버 식스": "6C", "클로버 세븐": "7C", "클로버 에잇": "8C", "클로버 나인": "9C", "클로버 텐": "10C",
    "클로버 킹": "KC", "클로버 퀸": "QC", "클로버 잭": "JC"
}
# -------- [4] 실시간 마이크로 PASS 카드 추출 --------
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
        language_code="ko-KR"  # 한국어
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=False
    )
    with sd.RawInputStream(samplerate=RATE, blocksize=CHUNK, dtype='int16',
                           channels=1, callback=callback):
        print("PASS할 카드를 말하세요 (예: '하트 세븐')")
        audio_generator = (audio_q.get() for _ in range(50))  # 5초 정도
        requests = (speech.StreamingRecognizeRequest(audio_content=chunk)
                    for chunk in audio_generator)
        responses = client.streaming_recognize(streaming_config, requests)
        for response in responses:
            for result in response.results:
                if result.is_final:
                    transcript = result.alternatives[0].transcript.strip().lower()
                    print("인식된 문장:", transcript)
                    for key, code in card_map.items():
                        if key in transcript:
                            print(f"인식된 카드 코드: {code}")
                            return code
                    print("카드 코드 매칭 실패")
                    return None
# -------- [5] YOLO 실행 + PASS 카드 무시 + 시리얼 전송 --------
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
                                symbol = label[-1].upper()  # C/H/D/S
                                print(f"[SEND] {symbol} ({label})")
                                ser.write(symbol.encode())
                            sent_cards.add(label)
                            time.sleep(0.6)
        # 시각화
        for x1, y1, x2, y2, label, conf in detections:
            color = (0, 0, 255) if label == skip_card else (0, 255, 0)
            cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, t=2, colorC=color)
            cvzone.putTextRect(img, f"{label} {conf:.2f}", (x1, y1 - 10), scale=1, thickness=2, colorT=color)
        if skip_card:
            cv2.putText(img, f"PASS: {skip_card}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        cv2.imshow("YOLOv8 카드 분류", img)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
# -------- [6] 실행부 --------
if __name__ == "__main__":
    yolo_model_path = "/Users/abitria/coding/python/yolov8s_playing_cards.pt"
    skip_card = get_card_code_from_mic()
    if skip_card:
        yolo_with_pass(skip_card, yolo_model_path)
    else:
        print("PASS 카드 인식 실패")