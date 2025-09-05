import os
import wave
import json
import queue
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from ultralytics import YOLO
import cv2, cvzone
import serial
import time

# -------- [1] 카드 클래스명 --------
classNames = [
    '10C','10D','10H','10S','2C','2D','2H','2S',
    '3C','3D','3H','3S','4C','4D','4H','4S',
    '5C','5D','5H','5S','6C','6D','6H','6S',
    '7C','7D','7H','7S','8C','8D','8H','8S',
    '9C','9D','9H','9S','AC','AD','AH','AS',
    'JC','JD','JH','JS','KC','KD','KH','KS',
    'QC','QD','QH','QS'
]

# -------- [2] 카드 이름 → 코드 매핑 --------
card_map = {
    "spade ace": "AS", "spade two": "2S", "spade three": "3S", "spade four": "4S", "spade five": "5S",
    "spade six": "6S", "spade seven": "7S", "spade eight": "8S", "spade nine": "9S", "spade ten": "10S",
    "spade jack": "JS", "spade queen": "QS", "spade king": "KS",
    "heart ace": "AH", "heart two": "2H", "heart three": "3H", "heart four": "4H", "heart five": "5H",
    "heart six": "6H", "heart seven": "7H", "heart eight": "8H", "heart nine": "9H", "heart ten": "10H",
    "heart jack": "JH", "heart queen": "QH", "heart king": "KH",
    "diamond ace": "AD", "diamond two": "2D", "diamond three": "3D", "diamond four": "4D", "diamond five": "5D",
    "diamond six": "6D", "diamond seven": "7D", "diamond eight": "8D", "diamond nine": "9D", "diamond ten": "10D",
    "diamond jack": "JD", "diamond queen": "QD", "diamond king": "KD",
    "club ace": "AC", "club two": "2C", "club three": "3C", "club four": "4C", "club five": "5C",
    "club six": "6C", "club seven": "7C", "club eight": "8C", "club nine": "9C", "club ten": "10C",
    "club jack": "JC", "club queen": "QC", "club king": "KC"
}

# -------- [3] 숫자 단어 → 정수 인덱스 --------
def extract_number(text):
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "twenty one": 21, "twenty two": 22,
        "twenty three": 23, "twenty four": 24, "twenty five": 25,
        "twenty six": 26, "twenty seven": 27, "twenty eight": 28,
        "twenty nine": 29, "thirty": 30, "thirty one": 31,
        "forty": 40, "forty two": 42, "fifty two": 52
    }
    text = text.strip().lower()
    if text.isdigit():
        return int(text)
    return number_words.get(text)

# -------- [4] 마이크 입력 → 숫자 인덱스 인식 --------
def recognize_number_from_mic(vosk_model_path):
    model = Model(vosk_model_path)
    recognizer = KaldiRecognizer(model, 16000)
    q = queue.Queue()

    def callback(indata, frames, time_, status):
    	q.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
        print("숫자를 말하세요 (예: 'eighteen')")
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                text = json.loads(recognizer.Result()).get("text", "")
                print("인식:", text)
                number = extract_number(text)
                if number and 1 <= number <= 52:
                    return number

# -------- [5] 인덱스 → WAV 파일 경로 (단일 폴더) --------
def get_wav_path_by_index(index, base_dir="/Users/abitria/coding/python/stt"):
    return os.path.join(base_dir, f"{index}.wav")

# -------- [6] WAV → 텍스트 --------
def stt_from_wav(wav_path, vosk_model_path):
    model = Model(vosk_model_path)
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    text = ""
    while True:
        data = wf.readframes(4000)
        if not data: break
        if rec.AcceptWaveform(data):
            text += " " + json.loads(rec.Result()).get("text", "")
    text += " " + json.loads(rec.FinalResult()).get("text", "")
    return text.strip().lower()

# -------- [7] 텍스트 → 카드코드 --------
def get_card_code_from_text(text):
    for key, code in card_map.items():
        if key in text:
            return code
    return None

# -------- [8] YOLO 실행 --------
def yolo_with_skip(skip_card, yolo_model_path, serial_port="/dev/tty.usbserial-0001", baud=115200):
    model = YOLO(yolo_model_path)
    ser = serial.Serial(serial_port, baud, timeout=1)
    cap = cv2.VideoCapture(0)
    cap.set(3, 720)
    cap.set(4, 480)
    ser.write((skip_card + "\n").encode())
    print(f"PASS 카드 전송됨 → {skip_card}")

    while True:
        success, img = cap.read()
        if not success:
            print("카메라 오류")
            break
        for r in model(img, stream=True):
            for box in r.boxes:
                cls = int(box.cls[0])
                label = classNames[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 0, 255) if label == skip_card else (0, 255, 0)
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, t=2, colorC=color)
                cvzone.putTextRect(img, f"{label} {conf:.2f}", (x1, y1 - 10), scale=1, thickness=2, colorT=color)
                if conf > 0.5 and label != skip_card:
                    symbol = label[-1]
                    print(f"[SEND] {symbol}")
                    ser.write(symbol.encode())
                    time.sleep(0.5)
                    cap.release()
                    cv2.destroyAllWindows()
                    ser.close()
                    return
        cv2.imshow("YOLO Card Detection", img)
        if cv2.waitKey(1) in [27, ord('q')]: break
    cap.release()
    cv2.destroyAllWindows()
    ser.close()

# -------- [9] 메인 실행 --------
if __name__ == "__main__":
    vosk_model_path = "/Users/abitria/coding/python/vosk-model-small-en-us-0.15"
    yolo_model_path = "/Users/abitria/coding/python/yolov8s_playing_cards.pt"
    wav_base_path = "/Users/abitria/coding/python/stt"  

    index = recognize_number_from_mic(vosk_model_path)
    print(f"인식된 숫자: {index}")

    wav_path = get_wav_path_by_index(index, base_dir=wav_base_path)
    print(f"해당 WAV 파일: {wav_path}")

    text = stt_from_wav(wav_path, vosk_model_path)
    print("STT 결과:", text)

    skip_card = get_card_code_from_text(text)
    if skip_card:
        print(f"PASS 카드 코드: {skip_card}")
        yolo_with_skip(skip_card, yolo_model_path)
    else:
        print("카드코드 매핑 실패")