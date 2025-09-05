from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import cv2, cvzone, serial, time, os, sys
import numpy as np
from threading import Thread
from ultralytics import YOLO
import speech_recognition as sr

# ==== 환경 변수 (Google STT 인증 키) ====
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/abitria/coding/python/stt.json"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== 모델 정의 ====
MODEL_CLASSNAMES = {
    "playing_cards": [...],  # 생략: 실제 카드 클래스명 리스트 복사 필요
    "beef": ['grade_1', 'grade_1p', 'grade_1pp', 'grade_2', 'grade_3'],
    "beverage": ['CocaColaKorea', 'DongwonFB','Haitaihtb','LotteChilsung','WoongjinFoods'],
    "recycle": ['plastic', 'glass', 'metal', 'cardboard', 'battery'],
}
LABEL_TO_PACKET = {
    "playing_cards": lambda label: label[-1].lower(),
    "beef": {'grade_1': 'h','grade_1p': 'c','grade_1pp': 's','grade_2': 'd'},
    "beverage": {'CocaColaKorea': 's','DongwonFB': 'c','Haitaihtb': 'h','LotteChilsung': 'd'},
    "recycle": {'plastic': 's','glass': 'c','metal': 'h','cardboard': 'd'},
}
MODEL_PATHS = {
    "playing_cards": "/Users/abitria/coding/python/yolov8s_playing_cards.pt",
    "beef": "/Users/abitria/coding/python/beef.pt",
    "beverage": "/Users/abitria/coding/python/beverage.pt",
    "recycle": "/Users/abitria/coding/python/recycle.pt",
}
DISPLAY_TO_CODE = {
    "트럼프 카드": "playing_cards",
    "고기등급": "beef",
    "음료수 종류": "beverage",
    "재활용품 분류": "recycle",
}
CODE_TO_DISPLAY = {v: k for k, v in DISPLAY_TO_CODE.items()}

SERIAL_PORT = os.environ.get("SERIAL_PORT", "/dev/tty.usbserial-0001")
_current_model_code: Optional[str] = None
_current_thread: Optional[Thread] = None
_stop_flag = False

def get_packet_char(model_name, label):
    mapper = LABEL_TO_PACKET[model_name]
    return mapper(label) if callable(mapper) else mapper.get(label, None)

def stt_worker():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("[STT] 음성 인식 대기 중...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    while not _stop_flag:
        with mic as source:
            try:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
                text = recognizer.recognize_google_cloud(audio, language='ko-KR')
                print(f"[STT] 인식됨: {text}")
                # 여기에 특정 명령어 처리 로직 추가 가능
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"[STT 오류] {e}")
                break

def yolo_worker(model_name):
    global _stop_flag
    model_path = MODEL_PATHS[model_name]
    classNames = MODEL_CLASSNAMES[model_name]
    model = YOLO(model_path)
    ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    is_cls = model.task == "classify"
    _stop_flag = False

    stt_thread = Thread(target=stt_worker, daemon=True)
    stt_thread.start()

    while not _stop_flag:
        success, img = cap.read()
        if not success:
            break
        h, w = img.shape[:2]
        left = w // 3
        right = w - w * 4 // 9
        masked = np.zeros_like(img)
        masked[:, left:right, :] = img[:, left:right, :]
        img_masked = masked

        if is_cls:
            results = model.predict(img_masked, device="mps")
            result = results[0]
            if hasattr(result, "probs") and result.probs is not None:
                idx = int(result.probs.top1)
                label = classNames[idx]
                conf = float(result.probs.data[idx])
                pkt = get_packet_char(model_name, label)
                print(f"[CLS] {label} ({conf:.2f}) → {pkt or 'PASS'}")
                if conf > 0.5 and pkt:
                    ser.write(pkt.encode())
                    time.sleep(0.7)
        else:
            detected_any = False
            for r in model(img_masked, stream=True):
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = classNames[cls]
                    conf = float(box.conf[0])
                    pkt = get_packet_char(model_name, label)
                    if conf > 0.5 and pkt:
                        print(f"[DET] {label} ({conf:.2f}) → {pkt}")
                        ser.write(pkt.encode())
                        time.sleep(0.7)
                        detected_any = True
                if detected_any:
                    break

        cv2.imshow(f"YOLO - {model_name}", img_masked)
        if cv2.waitKey(1) in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()
    ser.close()
    print("[END] 모델 중단됨")

class LoadModelReq(BaseModel):
    name: Optional[str]

@app.get("/v1/status")
def status():
    return {
        "model": CODE_TO_DISPLAY.get(_current_model_code),
        "running": _current_thread is not None and _current_thread.is_alive(),
        "serial_port": SERIAL_PORT
    }

@app.get("/v1/models")
def models():
    return {"models": [{"name": k} for k in DISPLAY_TO_CODE.keys()]}

@app.post("/v1/load_model")
def load_model(req: LoadModelReq):
    global _current_thread, _current_model_code, _stop_flag
    print(f"[DEBUG] 받은 모델 이름: {req.name}")
    if not req.name or req.name not in DISPLAY_TO_CODE:
        return {"ok": False, "error": f"unknown model name: {req.name}"}

    model_code = DISPLAY_TO_CODE[req.name]
    if _current_thread and _current_thread.is_alive():
        _stop_flag = True
        _current_thread.join()

    _current_model_code = model_code
    _current_thread = Thread(target=yolo_worker, args=(model_code,))
    _current_thread.start()
    return {"ok": True, "model": req.name}

@app.post("/v1/stop")
def stop():
    global _stop_flag, _current_thread
    _stop_flag = True
    if _current_thread:
        _current_thread.join()
    _current_thread = None
    return {"ok": True}
