# main.py
# pip install fastapi uvicorn ultralytics opencv-python pyserial pydantic pillow

import os, time, base64, io, cv2, serial
import numpy as np
from PIL import Image
from threading import Thread, Lock
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# (선택) 파이썬에서 Google STT도 쓸 거면 주석 해제
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/abitria/coding/python/stt.json"

# ======== 모델/경로/표기 ========
MODEL_CLASSNAMES = {
    "playing_cards": [
        '10C','10D','10H','10S','2C','2D','2H','2S',
        '3C','3D','3H','3S','4C','4D','4H','4S',
        '5C','5D','5H','5S','6C','6D','6H','6S',
        '7C','7D','7H','7S','8C','8D','8H','8S',
        '9C','9D','9H','9S','AC','AD','AH','AS',
        'JC','JD','JH','JS','KC','KD','KH','KS',
        'QC','QD','QH','QS'
    ],
    "beef":      ['grade_1','grade_1p','grade_1pp','grade_2','grade_3'],
    "beverage":  ['CocaColaKorea','DongwonFB','Haitaihtb','LotteChilsung','WoongjinFoods'],
    "recycle":   ['plastic','glass','metal','cardboard','battery'],
}
MODEL_PATHS = {
    "playing_cards": "/Users/abitria/coding/python/yolov8s_playing_cards.pt",
    "beef":          "/Users/abitria/coding/python/beef.pt",
    "beverage":      "/Users/abitria/coding/python/beverage.pt",
    "recycle":       "/Users/abitria/coding/python/recycle.pt",
}
DISPLAY_TO_CODE = {
    "트럼프 카드": "playing_cards",
    "고기등급":   "beef",
    "음료수 종류": "beverage",
    "재활용품 분류": "recycle",
}
CODE_TO_DISPLAY = {v: k for k, v in DISPLAY_TO_CODE.items()}
MODEL_ALIASES = {  # 웹/앱 혼용 대비 별칭
    "cards": "playing_cards",
}

def get_packet_char(model_name, label):
    if model_name == "playing_cards":
        return label[-1].lower()
    mapping = {
        "beef":     {'grade_1':'s','grade_1p':'c','grade_1pp':'h','grade_2':'d'},
        "beverage": {'CocaColaKorea':'s','DongwonFB':'c','Haitaihtb':'h','LotteChilsung':'d'},
        "recycle":  {'plastic':'s','glass':'c','metal':'h','cardboard':'d'},
    }
    return mapping.get(model_name, {}).get(label, None)

# ======== 보팅/쿨다운/장치 ========
VOTING_WINDOW_S = 3.0
START_THRES     = 0.15
REARM_DELAY_S   = 1.0

SERIAL_PORT = os.environ.get("SERIAL_PORT", "/dev/tty.usbserial-0001")
BAUDRATE    = 115200
CAM_INDEX   = 0
FRAME_W, FRAME_H = 720, 480

USE_ROI = True
ROI_L, ROI_R = 0.33, 1.0 - 4.0/9.0
DEVICE = "mps"  # Mac(MPS) / 아니면 "cpu"

def apply_roi(img: np.ndarray) -> np.ndarray:
    if not USE_ROI:
        return img
    h, w = img.shape[:2]
    left  = int(w * ROI_L)
    right = int(w * ROI_R)
    masked = np.zeros_like(img)
    masked[:, left:right, :] = img[:, left:right, :]
    return masked

# ======== 서버/스레드 상태 ========
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_current_model_code: Optional[str] = None
_current_thread: Optional[Thread]  = None
_stop_flag = False

# predict 전용 모델 캐시(오버레이용); worker와 별도 객체 사용
_predict_cache: Dict[str, YOLO] = {}
_predict_lock = Lock()

class LoadModelReq(BaseModel):
    # 앱/웹 둘 다 호환: name(한글) 또는 model(코드)
    name: Optional[str] = None
    model: Optional[str] = None

# ======== 유틸 ========
def _decode_data_url_to_ndarray(data_url: str) -> np.ndarray:
    content = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(content)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)

def _get_predict_model(model_code: str) -> YOLO:
    with _predict_lock:
        m = _predict_cache.get(model_code)
        if m is None:
            weights = MODEL_PATHS[model_code]
            if not os.path.exists(weights):
                raise FileNotFoundError(f"weights not found: {weights}")
            m = YOLO(weights)
            _predict_cache[model_code] = m
        return m

# ======== 작업 스레드(보팅 + 아두이노 전송) ========
def yolo_worker(model_code: str):
    global _stop_flag
    model_path = MODEL_PATHS[model_code]
    classnames = MODEL_CLASSNAMES[model_code]
    model = YOLO(model_path)

    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    vote_active = False
    vote_start  = 0.0
    votes_vec   = None
    next_rearm_time = 0.0

    print(f"[WORKER] start: {model_code}")
    while not _stop_flag:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01); continue

        img = apply_roi(frame)
        t_now = time.perf_counter()

        results = model.predict(img, device=DEVICE, verbose=False)
        r = results[0]

        if hasattr(r, "probs") and (r.probs is not None):
            probs = r.probs.data
            probs = probs.cpu().numpy() if hasattr(probs, "cpu") else np.asarray(probs)
            k = int(np.argmax(probs))
            p = float(probs[k])

            if (not vote_active) and (t_now >= next_rearm_time) and (p >= START_THRES):
                vote_active = True
                vote_start  = t_now
                votes_vec   = np.zeros_like(probs, dtype=np.float32)
                print("[VOTE] start (3s)")

            if vote_active:
                votes_vec += probs
                if (t_now - vote_start) >= VOTING_WINDOW_S:
                    final_k = int(np.argmax(votes_vec))
                    final_label = classnames[final_k]
                    pkt = get_packet_char(model_code, final_label)
                    if pkt:
                        ser.write(pkt.encode())   # ★ 한 글자만 전송
                        print(f"[SEND] {final_label} -> '{pkt}'")
                    else:
                        print(f"[SEND] skipped (no pkt for {final_label})")
                    vote_active = False
                    votes_vec   = None
                    next_rearm_time = time.perf_counter() + REARM_DELAY_S

    cap.release()
    ser.close()
    print("[WORKER] stopped]")

# ======== API ========
@app.get("/v1/models")
def models():
    return {"models": [{"name": n} for n in DISPLAY_TO_CODE.keys()]}

@app.get("/v1/status")
def status():
    return {
        "model": CODE_TO_DISPLAY.get(_current_model_code),
        "running": _current_thread is not None and _current_thread.is_alive(),
        "serial_port": SERIAL_PORT,
        "voting_window_s": VOTING_WINDOW_S,
        "rearm_delay_s": REARM_DELAY_S
    }

@app.post("/v1/load_model")
def load_model(req: LoadModelReq):
    global _stop_flag, _current_thread, _current_model_code

    # name(한글) 또는 model(코드) 허용 + 별칭 보정
    model_code: Optional[str] = None
    if req.name and req.name in DISPLAY_TO_CODE:
        model_code = DISPLAY_TO_CODE[req.name]
    elif req.model:
        code = MODEL_ALIASES.get(req.model, req.model)
        if code in MODEL_PATHS:
            model_code = code

    if not model_code:
        return {"ok": False, "error": f"unknown model: name={req.name}, model={req.model}"}

    # stop current
    if _current_thread and _current_thread.is_alive():
        _stop_flag = True
        _current_thread.join()

    # start new
    _stop_flag = False
    _current_model_code = model_code
    _current_thread = Thread(target=yolo_worker, args=(model_code,), daemon=True)
    _current_thread.start()
    return {"ok": True, "model": CODE_TO_DISPLAY.get(model_code, model_code)}

@app.post("/v1/stop")
def stop():
    global _stop_flag, _current_thread
    _stop_flag = True
    if _current_thread:
        _current_thread.join()
    _current_thread = None
    return {"ok": True}

# ★ 웹 오버레이용: dataURL(base64) 프레임 1회 추론 결과 반환
@app.post("/v1/predict")
async def predict(req: Request):
    if _current_model_code is None:
        return {"ok": False, "error": "no model running"}

    data = await req.json()
    img64 = data.get("image")
    if not img64:
        return {"ok": False, "error": "missing image"}

    img = _decode_data_url_to_ndarray(img64)
    m = _get_predict_model(_current_model_code)

    # worker와 충돌 줄이기 위해 잠깐 락
    with _predict_lock:
        res = m(img)
    r = res[0]
    preds: List[Dict[str, Any]] = []

    # detect 모델일 때
    if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
        names = getattr(r, 'names', None) or getattr(m, 'names', {})
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0]) if hasattr(b, 'conf') else None
            idx  = int(b.cls[0])   if hasattr(b, 'cls') else None
            label = names.get(idx, str(idx)) if isinstance(names, dict) else str(idx)
            preds.append({"label": label, "conf": conf, "box": [x1, y1, max(0, x2-x1), max(0, y2-y1)]})
    else:
        # classify 모델일 때: 중앙 더미 박스 + top1
        probs = getattr(r, 'probs', None)
        if probs is not None:
            k = int(probs.top1)
            conf = float(getattr(probs, 'top1conf', probs.data[k]))
            names = getattr(r, 'names', None) or getattr(m, 'names', {})
            label = names.get(k, str(k)) if isinstance(names, dict) else str(k)
            h, w = img.shape[:2]
            preds = [{"label": label, "conf": conf, "box": [int(w*0.3), int(h*0.3), int(w*0.4), int(h*0.4)]}]

    return {"ok": True, "model": _current_model_code, "preds": preds}

# ======== 정적(index.html, 이미지 등) 서빙 ========
# 같은 폴더에 index.html/이미지/PT 파일을 두면 /로 접근 가능
app.mount("/", StaticFiles(directory=".", html=True), name="static")


##############################################################################################################
#python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload