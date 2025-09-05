import cv2
import threading
import queue
import time
import json
import pyaudio
from vosk import Model, KaldiRecognizer
from ultralytics import YOLO
# ------------ [1] YOLO 모델 로딩 ------------
model = YOLO("yolov8s_playing_cards.pt")  # 카드 인식용 YOLOv8 모델
# ------------ [2] VOSK Giga 로딩 ------------
vosk_model_path = "vosk-model-en-us-0.42-gigaspeech"
vosk_model = Model(vosk_model_path)
recognizer = KaldiRecognizer(vosk_model, 16000)
# ------------ [3] 음성 인식용 오디오 설정 ------------
audio_q = queue.Queue()
stop_flag = threading.Event()
def audio_capture():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=16000,
                     input=True,
                     frames_per_buffer=8000)
    stream.start_stream()
    print("STT Listening...")
    while not stop_flag.is_set():
        data = stream.read(4000, exception_on_overflow=False)
        audio_q.put(data)
    stream.stop_stream()
    stream.close()
    pa.terminate()
def stt_thread():
    buffer = b''
    while not stop_flag.is_set():
        if not audio_q.empty():
            buffer += audio_q.get()
            if recognizer.AcceptWaveform(buffer):
                res = json.loads(recognizer.Result())
                text = res.get('text', '')
                if text:
                    print(f"[STT]: {text}")
                    buffer = b''  # reset buffer
# ------------ [4] YOLO 영상 처리 쓰레드 ------------
def yolo_thread():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라 접근 실패")
        return
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO 카드 인식", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()
            break
    cap.release()
    cv2.destroyAllWindows()
# ------------ [5] 메인 실행 ------------
try:
    t1 = threading.Thread(target=audio_capture)
    t2 = threading.Thread(target=stt_thread)
    t3 = threading.Thread(target=yolo_thread)
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()
except KeyboardInterrupt:
    stop_flag.set()
    print("종료 중...")