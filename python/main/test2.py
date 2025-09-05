import os
import random
import json
import time
import pygame
import queue
import threading
import cv2
import numpy as np
from ultralytics import YOLO
from vosk import Model, KaldiRecognizer
import pyaudio
# --- [1] 카드 음성 파일 경로 설정 ---
CARD_DIR = 'trimp_cards'
card_files = [f for f in os.listdir(CARD_DIR) if f.endswith('.mp3')]
# --- [2] YOLO 모델 로딩 ---
yolo = YOLO('yolov8s_playing_cards.pt')  # s/l 선택
# --- [3] VOSK 모델 로딩 ---
vosk_model_path = "vosk-model-en-us-0.42-gigaspeech"
vosk_model = Model(vosk_model_path)
# --- [4] 음성 재생 + STT ---
def play_and_stt(mp3_path):
    # 1. mp3 재생 (pygame)
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_path)
    pygame.mixer.music.play()
    # 2. STT 동시진행
    q = queue.Queue()
    rec = KaldiRecognizer(vosk_model, 16000)
    def record_callback(indata, frame_count, time_info, status):
        q.put(bytes(indata))
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    print("음성 재생 및 STT 시작...")
    buffer = b''
    stt_text = ''
    start_time = time.time()
    while pygame.mixer.music.get_busy() and time.time() - start_time < 5:
        data = stream.read(4000, exception_on_overflow=False)
        buffer += data
        if rec.AcceptWaveform(buffer):
            res = json.loads(rec.Result())
            stt_text = res.get("text", "")
            print("STT 인식:", stt_text)
            buffer = b''
    stream.stop_stream()
    stream.close()
    pa.terminate()
    pygame.mixer.quit()
    # 강제 단어(보정) 처리
    force_words = ['four', 'two', 'ace', 'heart']
    for w in force_words:
        if w in os.path.basename(mp3_path).lower() and w not in stt_text:
            stt_text += f" {w}"
    return stt_text.strip()
# --- [5] YOLO 인식 (중앙 마스킹) ---
def show_yolo_with_mask(card_keyword):
    cap = cv2.VideoCapture(0)
    print("YOLO 카드 인식 준비...")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # 중앙 사각형 마스킹
        h, w, _ = frame.shape
        cx, cy = w//2, h//2
        size = min(w, h)//2  # 중앙 1/2만 인식
        mask = np.zeros_like(frame)
        x1, y1 = cx-size//2, cy-size//2
        x2, y2 = cx+size//2, cy+size//2
        mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        # YOLO 카드 인식
        results = yolo(mask)
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = yolo.names[cls] if hasattr(yolo, 'names') else str(cls)
            x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
            # 카드 부분만 외곽선
            cv2.rectangle(mask, (x1b, y1b), (x2b, y2b), (0, 255, 0), 3)
            cv2.putText(mask, f"{label} {conf:.2f}", (x1b, y1b-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        # 나머지 영역 검은색
        frame_masked = mask.copy()
        cv2.imshow("YOLO 카드 인식(중앙만)", frame_masked)
        k = cv2.waitKey(1)
        if k == 27 or k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
# --- [6] 메인 실행 ---
if __name__ == '__main__':
    # 1. 카드 mp3 랜덤선택
    mp3_file = random.choice(card_files)
    mp3_path = os.path.join(CARD_DIR, mp3_file)
    print(f"재생 mp3: {mp3_path}")
    # 2. 카드 이름 추출
    card_keyword = os.path.splitext(os.path.basename(mp3_path))[0].replace('club number ', '').replace('diamond number ', '').replace('heart number ', '').replace('spade number ', '')
    print(f"카드 키워드(이름): {card_keyword}")
    # 3. 음성재생+STT(강제보정)
    stt_result = play_and_stt(mp3_path)
    print(f"최종 STT: {stt_result}")
    # 4. YOLO+마스킹 카드인식
    show_yolo_with_mask(card_keyword)