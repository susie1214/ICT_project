# main_green.py
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
MODEL_ALIASES = { "cards": "playing_cards" }  # 웹/앱 혼용 대비

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

SERIAL_PORT = os.environ.get("SERIAL_PORT", "/dev/cu.usbserial-0001")
BAUDRATE    = 115200
CAM_INDEX   = 0
FRAME_W, FRAME_H = 720, 480

USE_ROI = True
ROI_L, ROI_R = 0.33, 1.0 - 4.0/9.0
#ROI_T, ROI_B = 0.10, 0.90
DEVICE = "mps"  # Mac(MPS) / 아니면 "cpu"

#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
# ======== 초록 배경 무시(ON) + 자동보정 ========
IGNORE_GREEN   = False
PROTECT_CENTER = False
AUTO_GREEN     = False
GREEN_LOWER    = np.array([35, 40, 40],  dtype=np.uint8)
GREEN_UPPER    = np.array([85, 255, 255], dtype=np.uint8)
GREEN_BOX_MAX_RATIO = 0.60  # 박스 내부 초록비율이 60% 이상이면 무시
_auto_green_done = False

_kernel5 = np.ones((5,5), np.uint8)
_green_lock = Lock()

def compute_green_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    with _green_lock:
        lo = GREEN_LOWER.copy(); hi = GREEN_UPPER.copy()
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _kernel5, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel5, 1)
    if PROTECT_CENTER:
        h, w = img_bgr.shape[:2]
        cx0, cx1 = int(w*0.30), int(w*0.70)
        cy0, cy1 = int(h*0.15), int(h*0.85)
        mask[cy0:cy1, cx0:cx1] = 0
    return mask

def remove_green_background(img_bgr):
    if not IGNORE_GREEN:
        return img_bgr
    mask = compute_green_mask(img_bgr)
    out = img_bgr.copy()
    out[mask > 0] = (128, 128, 128)
    return out

def autocalib_green_from_edges(frame_bgr):
    h, w = frame_bgr.shape[:2]
    band = np.vstack([frame_bgr[0:int(0.15*h), :], frame_bgr[int(0.85*h):, :]])
    hsv  = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    h_med = np.median(hsv[:,:,0]); s_med = np.median(hsv[:,:,1])
    lo = np.array([max(0,   int(h_med-10)), max(0, int(s_med-40)), 30], dtype=np.uint8)
    hi = np.array([min(179, int(h_med+10)), 255,                     255], dtype=np.uint8)
    return lo, hi

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
    name: Optional[str] = None  # 한글
    model: Optional[str] = None # 코드/별칭

# ======== 유틸 ========
def _decode_data_url_to_ndarray(data_url: str) -> np.ndarray:
    content = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(content)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)[:, :, ::-1]  # RGB->BGR (cv2)

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

# MCU 로그/READY 동기화
def drain_logs(ser):
    try:
        n = ser.in_waiting
        if n:
            data = ser.read(n)
            if data:
                try: print("[MCU]", data.decode(errors="ignore").strip())
                except Exception: print("[MCU bytes]", data)
    except Exception:
        pass

def wait_ready(ser, timeout=10.0):
    t0 = time.time()
    buf = b""
    ser.timeout = 0.1
    while time.time() - t0 < timeout:
        drain_logs(ser)
        n = ser.in_waiting
        if n:
            buf += ser.read(n)
            if b"READY" in buf:
                return True
        time.sleep(0.02)
    return False

# ======== 작업 스레드(보팅 + 아두이노 전송) ========
def yolo_worker(model_code: str):
    global _stop_flag, _auto_green_done, GREEN_LOWER, GREEN_UPPER

    model_path = MODEL_PATHS[model_code]
    classnames = MODEL_CLASSNAMES[model_code]
    model = YOLO(model_path)

    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
    time.sleep(0.2)
    drain_logs(ser)

    # LCD 모델 전환: 1=CARD, 2=RECYCLE, 3=BEVERAGE, 4=BEEF
    select_map = {'playing_cards':'1', 'recycle':'2', 'beverage':'3', 'beef':'4'}
    sel = select_map.get(model_code)
    if sel:
        ser.write(sel.encode()); time.sleep(0.05)
        ser.write(b'R');         time.sleep(0.05)
        drain_logs(ser)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    vote_active = False
    vote_start  = 0.0
    votes_vec   = None
    next_rearm_time = 0.0
    is_classify = (getattr(model, "task", None) == "classify")

    print(f"[WORKER] start: {model_code}")
    while not _stop_flag:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01); continue

        # 자동 초록 보정(1회)
        if AUTO_GREEN and not _auto_green_done:
            lo, hi = autocalib_green_from_edges(frame)
            with _green_lock:
                GREEN_LOWER[:], GREEN_UPPER[:] = lo, hi
            _auto_green_done = True
            print(f"[GREEN] AUTO calib LOWER={lo.tolist()} UPPER={hi.tolist()}")

        roi = apply_roi(frame)
        gmask = compute_green_mask(roi) if IGNORE_GREEN else None
        img  = remove_green_background(roi)
        t_now = time.perf_counter()

        r = model.predict(img, device=DEVICE, verbose=False)[0]

        if is_classify and getattr(r, "probs", None) is not None:
            probs = r.probs.data
            probs = probs.cpu().numpy() if hasattr(probs, "cpu") else np.asarray(probs)
            k = int(np.argmax(probs)); p = float(probs[k])

            if (not vote_active) and (t_now >= next_rearm_time) and (p >= START_THRES):
                vote_active = True; vote_start = t_now
                votes_vec = np.zeros_like(probs, dtype=np.float32)
                print("[VOTE] start")

            if vote_active:
                votes_vec += probs
                if (t_now - vote_start) >= VOTING_WINDOW_S:
                    final_k = int(np.argmax(votes_vec))
                    final_label = classnames[final_k]
                    pkt = get_packet_char(model_code, final_label)
                    if pkt:
                        print("[SER] wait READY…")
                        if wait_ready(ser, timeout=10.0):
                            ser.write(pkt.encode())
                            print(f"[SEND] {final_label} -> '{pkt}'")
                        else:
                            print("[WARN] READY timeout → skip send")
                    else:
                        print(f"[SEND] skipped (no pkt for {final_label})")
                    vote_active = False; votes_vec = None
                    next_rearm_time = time.perf_counter() + REARM_DELAY_S

        else:
            # 탐지: 박스 conf 누적(초록비율 높으면 무시)
            n_classes = len(classnames)
            frame_scores = np.zeros(n_classes, dtype=np.float32)
            if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
                h, w = img.shape[:2]
                for b in r.boxes:
                    cls = int(b.cls[0])
                    if not (0 <= cls < n_classes): continue
                    conf = float(b.conf[0]) if hasattr(b,"conf") else 1.0
                    # 초록 비율 필터
                    if gmask is not None:
                        x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                        x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
                        if x2<=x1 or y2<=y1: continue
                        crop = gmask[y1:y2, x1:x2]
                        green_ratio = float((crop>0).mean()) if crop.size else 0.0
                        if green_ratio >= GREEN_BOX_MAX_RATIO:
                            continue
                    frame_scores[cls] += conf

            # 시작/투표/전송
            start_ok = frame_scores.max() > 0.0
            if (not vote_active) and (t_now >= next_rearm_time) and start_ok:
                vote_active = True; vote_start = t_now
                votes_vec = np.zeros_like(frame_scores, dtype=np.float32)
                print("[VOTE] start(det)")

            if vote_active:
                votes_vec += frame_scores
                if (t_now - vote_start) >= VOTING_WINDOW_S:
                    final_k = int(np.argmax(votes_vec))
                    final_label = classnames[final_k]
                    pkt = get_packet_char(model_code, final_label)
                    if pkt:
                        print("[SER] wait READY…")
                        if wait_ready(ser, timeout=10.0):
                            ser.write(pkt.encode())
                            print(f"[SEND] {final_label} -> '{pkt}'")
                        else:
                            print("[WARN] READY timeout → skip send")
                    else:
                        print(f"[SEND] skipped (no pkt for {final_label})")
                    vote_active = False; votes_vec = None
                    next_rearm_time = time.perf_counter() + REARM_DELAY_S

    cap.release()
    ser.close()
    print("[WORKER] stopped")

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
    global _stop_flag, _current_thread, _current_model_code, _auto_green_done

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

    # green calib 리셋
    _auto_green_done = False

    # start new
    _stop_flag = False
    _current_model_code = model_code
    _current_thread = Thread(target=yolo_worker, args=(model_code,), daemon=True)
    _current_thread.start()
    return {"ok": True, "model": CODE_TO_DISPLAY.get(model_code, model_code)}

# 호환용 별칭
@app.post("/v1/switch_model")
def switch_model_alias(req: LoadModelReq):
    return load_model(req)

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

    bgr = _decode_data_url_to_ndarray(img64)
    if AUTO_GREEN and not _auto_green_done:
        lo, hi = autocalib_green_from_edges(bgr)
        with _green_lock:
            GREEN_LOWER[:], GREEN_UPPER[:] = lo, hi

    roi = apply_roi(bgr)
    gmask = compute_green_mask(roi) if IGNORE_GREEN else None
    img  = remove_green_background(roi)

    m = _get_predict_model(_current_model_code)
    with _predict_lock:
        r = m(img)[0]

    preds: List[Dict[str, Any]] = []
    names = getattr(r, 'names', None) or getattr(m, 'names', {})
    # detect
    if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
        h, w = img.shape[:2]
        for b in r.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0]) if hasattr(b, 'conf') else None
            idx  = int(b.cls[0])   if hasattr(b, 'cls') else None
            if gmask is not None:
                xx1=max(0,x1); yy1=max(0,y1); xx2=min(w,x2); yy2=min(h,y2)
                if xx2>xx1 and yy2>yy1:
                    crop = gmask[yy1:yy2, xx1:xx2]
                    if crop.size and float((crop>0).mean()) >= GREEN_BOX_MAX_RATIO:
                        continue
            label = names.get(idx, str(idx)) if isinstance(names, dict) else str(idx)
            preds.append({"label": label, "conf": conf, "box": [x1, y1, max(0,x2-x1), max(0,y2-y1)]})
    else:
        # classify: 중앙 더미박스 + top1
        probs = getattr(r, 'probs', None)
        if probs is not None:
            k = int(probs.top1)
            conf = float(getattr(probs, 'top1conf', probs.data[k]))
            label = names.get(k, str(k)) if isinstance(names, dict) else str(k)
            h, w = img.shape[:2]
            preds = [{"label": label, "conf": conf, "box": [int(w*0.3), int(h*0.3), int(w*0.4), int(h*0.4)]}]

    return {"ok": True, "model": _current_model_code, "preds": preds}

# ======== 정적(index.html, 이미지 등) 서빙 ========
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# Run:
# python3 -m uvicorn main_green:app --host 0.0.0.0 --port 8000 --reload
