from vosk import Model, KaldiRecognizer
import wave, json, serial, time
from ultralytics import YOLO
import cv2, cvzone
import threading
# -------- [1] 카드 클래스명 (YOLO 학습 순서와 일치) --------
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
# -------- [3] STT: wav 파일 → 텍스트 --------
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
# -------- [4] 텍스트 → YOLO 클래스명 코드 --------
def get_card_code_from_text(text):
    for key, code in card_map.items():
        if key in text:
            return code
    return None
# -------- [5] YOLO 탐지 + 카드 분류 + PASS카드 무시 + LCD 출력을 위한 카드코드 전송 --------
def yolo_with_skip(skip_card, yolo_model_path, serial_port="/dev/tty.usbserial-0001", baud=115200):
    model = YOLO(yolo_model_path)
    ser = serial.Serial(serial_port, baud, timeout=1)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # LCD에 패스카드 정보 전송
    ser.write((skip_card + "\n").encode())
    print(f"PASS 카드 전송됨 → {skip_card}")
    detected = False
    while not detected:
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
                cvzone.putTextRect(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                                   scale=1, thickness=2, colorT=color)
                if conf > 0.5 and label != skip_card:
                    symbol = label[-1]  # "6H" → "H"
                    print(f"[SEND] {symbol}")
                    ser.write(symbol.encode())
                    time.sleep(0.5)
                    detected = True
                    # break
            if detected:
                # break
                pass
        cv2.imshow("YOLO Card Detection", img)
        key = cv2.waitKey(1)
        if key in [27, ord('q')]:  # ESC나 q키 누르면 수동종료
            detected = True
            break
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
# -------- [6] 엔터 입력시 종료 구현 --------
stop_flag = False
def wait_for_exit():
    global stop_flag
    input("종료하려면 엔터를 누르세요...\n")
    stop_flag = True

    
if __name__ == "__main__":
    wav_path = "/Users/abitria/coding/python/voice_input.wav"
    vosk_model_path = "/Users/abitria/coding/python/vosk-model-small-en-us-0.15"
    yolo_model_path = "/Users/abitria/coding/python/yolov8s_playing_cards.pt"
    # 1. wav 파일 STT → PASS 카드코드 추출
    text = stt_from_wav(wav_path, vosk_model_path)
    print("STT 결과:", text)
    skip_card = get_card_code_from_text(text)
    if not skip_card:
        print("카드 인식 실패 → 종료")
    else:
        print(f"PASS 카드 코드: {skip_card}")
        # 엔터 누를 때까지 반복 분류
        exit_thread = threading.Thread(target=wait_for_exit)
        exit_thread.daemon = True
        exit_thread.start()
        while not stop_flag:
            yolo_with_skip(skip_card, yolo_model_path)
            if not stop_flag:
                print("카드가 분류되었습니다. (계속 반복)\n종료하려면 엔터를 누르세요.")
        print("프로그램을 종료합니다.")
