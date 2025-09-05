import threading
import queue
import time
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from ultralytics import YOLO
import cv2, cvzone
import serial

classNames = [
    '10C','10D','10H','10S','2C','2D','2H','2S',
    '3C','3D','3H','3S','4C','4D','4H','4S',
    '5C','5D','5H','5S','6C','6D','6H','6S',
    '7C','7D','7H','7S','8C','8D','8H','8S',
    '9C','9D','9H','9S','AC','AD','AH','AS',
    'JC','JD','JH','JS','KC','KD','KH','KS',
    'QC','QD','QH','QS'
]
# -------- [2] 음성 문자열 → 카드코드 매핑 --------
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
pass_card_code = ""
pass_card_lock = threading.Lock()
stop_flag = threading.Event()
# -------- [2] 마이크 1회 STT 인식 --------
def get_card_code_from_stt_once(vosk_model_path):
    model = Model(vosk_model_path)
    recognizer = KaldiRecognizer(model, 16000)
    q = queue.Queue()
    def callback(indata, frames, time_, status):
        q.put(bytes(indata))
    print("패스카드 이름을 마이크에 말하세요(5초 이내)...")
    with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype='int16', channels=1, callback=callback):
        buffer = b''
        while True:
            while not q.empty():
                buffer += q.get()
                if recognizer.AcceptWaveform(buffer):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").lower()
                    print(f"[STT] 인식: {text}")
                    for key, code in card_map.items():
                        if key in text:
                            print(f"PASS 카드가 {key}({code})로 변경됨!")
                            return code
                    buffer = b''
            time.sleep(0.05)
    print("인식 실패(시간 초과).")
    return None
# -------- [3] YOLO 분류 스레드 --------
def yolo_thread(yolo_model_path, serial_port="/dev/tty.usbserial-0001", baud=115200):
    model = YOLO(yolo_model_path)
    ser = serial.Serial(serial_port, baud, timeout=1)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    last_pass = ""
    while not stop_flag.is_set():
        with pass_card_lock:
            now_pass = pass_card_code
        if now_pass != last_pass and now_pass != "":
            ser.write((now_pass + "\n").encode())
            print(f"LCD PASS 카드 전송됨 → {now_pass}")
            last_pass = now_pass
        ret, img = cap.read()
        if not ret:
            continue
        for r in model(img, stream=True):
            for box in r.boxes:
                cls = int(box.cls[0])
                label = classNames[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 0, 255) if label == now_pass else (0, 255, 0)
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, t=2, colorC=color)
                cvzone.putTextRect(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                                   scale=1, thickness=2, colorT=color)
                if conf > 0.5 and now_pass != "" and label != now_pass:
                    symbol = label[-1]
                    print(f"[SEND] {symbol}")
                    ser.write(symbol.encode())
                    time.sleep(0.5)
        cv2.imshow("YOLO Card Detection", img)
        key = cv2.waitKey(1)
        if key in [27, ord('q')]:
            stop_flag.set()
            break
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
# -------- [4] 엔터 입력 → STT 활성화 반복 --------
def pass_card_input_thread(vosk_model_path):
    global pass_card_code
    # 1. 처음 시작할 때 자동 1회 STT
    print("\n--- 프로그램 시작: PASS카드 음성으로 먼저 등록 ---")
    code = None
    while code is None:
        code = get_card_code_from_stt_once(vosk_model_path)
        if code:
            with pass_card_lock:
                pass_card_code = code
        else:
            print("다시 시도하세요(마이크에 정확히 말하기)!")
    print(f"초기 PASS카드: {pass_card_code}\n")
    # 2. 이후 반복: 엔터 입력 시만 STT 실행
    while not stop_flag.is_set():
        user = input("PASS카드 변경하려면 Enter를 누르세요(종료는 q 입력): ")
        if user.strip().lower() == "q":
            stop_flag.set()
            break
        code = get_card_code_from_stt_once(vosk_model_path)
        if code:
            with pass_card_lock:
                pass_card_code = code
# -------- [5] 메인 실행 --------
if __name__ == "__main__":
    vosk_model_path = "/Users/abitria/coding/python/vosk-model-en-us-0.42-gigaspeech"
    yolo_model_path = "/Users/abitria/coding/python/yolov8s_playing_cards.pt"
    serial_port = "/dev/tty.usbserial-0001"
    t1 = threading.Thread(target=yolo_thread, args=(yolo_model_path, serial_port))
    t1.daemon = True
    t1.start()
    pass_card_input_thread(vosk_model_path)  # 메인에서 초기1회 + 반복 엔터 입력 처리
    stop_flag.set()
    print("전체 프로그램 종료")