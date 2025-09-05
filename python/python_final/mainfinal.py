# main.py  (TOP OF FILE)
# --- Quiet header: hide urllib3 warning + silence OpenCV logs ---
import os, warnings
# 1) urllib3 · LibreSSL 경고 숨기기
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:urllib3")
# 2) OpenCV 자체 로그 끄기 (AVFoundation 관련 메시지 억제)
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"   # must be set BEFORE importing cv2
import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass
# ----------------------------------------------------------------
# 이후 일반 import (중복 없이)
import time, base64, io, serial, asyncio
import numpy as np
from PIL import Image
from threading import Thread, Lock
from typing import Optional, Any, Dict, List
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO


# ================== 모델/경로/표기 ==================
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
MODEL_ALIASES = { "cards": "playing_cards" }

def get_packet_char(model_name: str, label: str) -> Optional[str]:
    if model_name == "playing_cards":
        return label[-1].lower() if label else None
    mapping = {
        "beef":     {'grade_1':'s','grade_1p':'c','grade_1pp':'h','grade_2':'d'},
        "beverage": {'CocaColaKorea':'s','DongwonFB':'c','Haitaihtb':'h','LotteChilsung':'d'},
        "recycle":  {'plastic':'s','glass':'c','metal':'h','cardboard':'d'},
    }
    return mapping.get(model_name, {}).get(label, None)

# ================== 데모/환경 ==================
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "playing_cards")
DEMO_CARDS_ONLY = os.getenv("DEMO_CARDS_ONLY", "0") == "1"   # 1이면 카드만 전송
def _can_send_for(model_code: str) -> bool:
    return (not DEMO_CARDS_ONLY) or (model_code == "playing_cards")
CLEAR_FRAMES = 6 #10
HOLD_KEEP_LABEL_MIN_CONF = 0.35   # 같은 라벨을 '아직 보인다'로 인정할 최소 conf(분류용)
RESEND_MIN_GAP_S = 3.5
# ================== 보팅/쿨다운/장치 ==================
VOTING_WINDOW_S = 2
START_THRES     = 0.15
REARM_DELAY_S   = 0

SERIAL_PORT = os.environ.get("SERIAL_PORT", "/dev/tty.usbserial-0001")
BAUDRATE    = 115200
CAM_INDEX   = 0
FRAME_W, FRAME_H = 720, 480

USE_ROI = True
ROI_L, ROI_R = 0.27, 0.6
TOP_MASK_FRAC = 0    # ← 위쪽 20% 블랙 마스킹
DEVICE = "mps"  # Mac

# === add near other utils ===
def names_from_model(model, result=None):
    """Ultralytics 모델/결과에서 클래스 이름을 안전하게 추출."""
    srcs = []
    if result is not None:
        srcs.append(getattr(result, "names", None))
    srcs.append(getattr(model, "names", None))
    for n in srcs:
        if isinstance(n, dict):   # {idx: name}
            return [n[i] for i in sorted(n)]
        if isinstance(n, (list, tuple)):
            return list(n)
    # fallback: 길이만 맞춰 더미 이름
    nc = getattr(getattr(model, "model", None), "nc", 0) or 0
    return [str(i) for i in range(nc)]

def apply_roi(img: np.ndarray) -> np.ndarray:
    """
    좌우 ROI만 살리고 나머지 컬럼은 0(검정)으로 마스킹.
    + 상단 TOP_MASK_FRAC(예: 20%) 행 전체를 0으로 마스킹.
    """
    if not USE_ROI:
        # 상단 마스킹만 적용하고 싶으면 아래 두 줄만 사용
        if TOP_MASK_FRAC > 0:
            h, w = img.shape[:2]
            top_h = int(h * TOP_MASK_FRAC)
            if top_h > 0:
                img = img.copy()
                img[:top_h, :, :] = 0
        return img

    h, w = img.shape[:2]
    L, R = int(w * ROI_L), int(w * ROI_R)

    masked = np.zeros_like(img)
    masked[:, L:R, :] = img[:, L:R, :]

    # 상단 20% 마스킹
    if TOP_MASK_FRAC > 0:
        top_h = int(h * TOP_MASK_FRAC)
        if top_h > 0:
            masked[:top_h, :, :] = 0

    return masked

# ================== 서버/상태 ==================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_current_model_code: Optional[str] = None
_current_thread: Optional[Thread]  = None
_stop_flag = False

# /v1/predict용 캐시 (웹캠 캡쳐 방식 유지할 때 사용)
_predict_cache: Dict[str, YOLO] = {}
_predict_lock = Lock()

# ★ MJPEG 스트리밍용 최신 프레임 버퍼
_last_jpeg: Optional[bytes] = None
_frame_lock = Lock()

class LoadModelReq(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None

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

# ================== 워커(카메라+보팅+오버레이+시리얼) ==================
def yolo_worker(model_code: str, send_enabled: bool):
    global _stop_flag, _last_jpeg
    model_path = MODEL_PATHS[model_code]
    model = YOLO(model_path)

    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    vote_active = False
    vote_start  = 0.0
    votes_vec   = None
    next_rearm  = 0.0

    names = None
    is_classify = (getattr(model, "task", None) == "classify")
    print(f"[WORKER] start: {model_code}, task={getattr(model,'task',None)} (send_enabled={send_enabled})")

    # ★ 중복 전송 방지 상태
    last_sent_label: Optional[str] = None
    last_sent_time: float = 0.0
    low_frames = 0  # '최근 보냈던 라벨'이 안 보이는 프레임 누적

    while not _stop_flag:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 0)
        img  = apply_roi(frame)
        tnow = time.perf_counter()

        res = model.predict(img, device=DEVICE, verbose=False)[0]
        if names is None:
            names = names_from_model(model, res)
            print(f"[INFO] class names loaded (n={len(names)})")

        # ---- 현재 프레임의 "최선 라벨/스코어"를 뽑아 둠
        best_label = None
        best_score = 0.0
        frame_scores = None
        start_ok = False

        if is_classify and getattr(res, "probs", None) is not None:
            probs = res.probs.data
            probs = probs.cpu().numpy() if hasattr(probs, "cpu") else np.asarray(probs)
            if probs.shape[-1] != len(names):
                names = names_from_model(model, res)
            frame_scores = probs.astype(np.float32)
            k = int(np.argmax(probs)); best_score = float(probs[k])
            best_label = names[k] if k < len(names) else str(k)
            start_ok = (best_score >= START_THRES)
        else:
            n_classes = len(names) if names else (getattr(getattr(model, "model", None), "nc", 0) or 0)
            frame_scores = np.zeros(n_classes, dtype=np.float32)
            if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    cls = int(b.cls[0]); conf = float(b.conf[0]) if hasattr(b, "conf") else 1.0
                    if 0 <= cls < n_classes:
                        frame_scores[cls] += conf
                k = int(np.argmax(frame_scores)); best_score = float(frame_scores[k])
                best_label = names[k] if k < len(names) else str(k)
                start_ok = (best_score > 0.0)

        # ---- 화면 표시용 힌트
        hint = "No boxes" if best_label is None else (f"{best_label} ({best_score:.2f})" if is_classify else f"{best_label} (+{best_score:.2f})")

        # ---- 같은 라벨 '여전히 보임' 판단(분류 기준: conf 유지)
        same_item_visible = False
        if last_sent_label is not None and best_label is not None and best_label == last_sent_label:
            thr = max(HOLD_KEEP_LABEL_MIN_CONF, START_THRES)
            same_item_visible = (best_score >= thr)

        if same_item_visible:
            low_frames = 0
        else:
            # 최근 보냈던 라벨이 지금은 안 보임 → 누적
            if last_sent_label is not None:
                low_frames += 1
                if low_frames >= CLEAR_FRAMES:
                    print("[HOLD] cleared (object gone) → allow re-send next time")
                    last_sent_label = None
                    low_frames = 0

        # ---- 같은 라벨이 잡혀 있는 동안은 아예 투표 시작 금지
        allow_vote_on_label = True
        if last_sent_label is not None and best_label == last_sent_label:
            allow_vote_on_label = False

        # ---- 투표 상태머신
        if (not vote_active) and (tnow >= next_rearm) and start_ok and allow_vote_on_label:
            vote_active = True
            vote_start  = tnow
            votes_vec   = np.zeros_like(frame_scores, dtype=np.float32)
            print("[VOTE] start")

        if vote_active:
            if votes_vec.shape != frame_scores.shape:
                votes_vec = np.zeros_like(frame_scores, dtype=np.float32)
            votes_vec += frame_scores
            if (tnow - vote_start) >= VOTING_WINDOW_S:
                final_k = int(np.argmax(votes_vec))
                final_label = names[final_k] if final_k < len(names) else str(final_k)

                # ★ 최소 시간 간격 + 같은 라벨 보류 체크
                if final_label == last_sent_label:
                    print(f"[SKIP] duplicate while same item present: {final_label}")
                elif (tnow - last_sent_time) < RESEND_MIN_GAP_S:
                    print(f"[SKIP] min-gap {RESEND_MIN_GAP_S:.1f}s not reached")
                else:
                    pkt = get_packet_char(model_code, final_label)
                    if pkt:
                        if send_enabled:
                            ser.write(pkt.encode())
                            print(f"[SEND] {final_label} -> '{pkt}'")
                        else:
                            print(f"[SKIP SEND] demo ({final_label} -> '{pkt}')")
                        last_sent_label = final_label
                        last_sent_time  = tnow
                        low_frames = 0
                    else:
                        print(f"[SEND] skipped (no pkt for '{final_label}')")

                vote_active = False
                votes_vec   = None
                next_rearm  = time.perf_counter() + REARM_DELAY_S

        # ---- 오버레이 & 전송(MJPEG)
        disp = img.copy()
        h, w = disp.shape[:2]
        L, R = int(w*ROI_L), int(w*ROI_R); top = int(h*TOP_MASK_FRAC)
        cv2.line(disp, (L,0), (L,h), (0,200,255), 2)
        cv2.line(disp, (R,0), (R,h), (0,200,255), 2)
        cv2.line(disp, (0,top), (w,top), (140,255,140), 2)
        cd = max(0.0, next_rearm - tnow)
        cv2.putText(disp, hint, (12,34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        cv2.putText(disp, f"vote:{'ON' if vote_active else 'OFF'}  cd:{cd:.1f}s",
                    (12,66), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,255,200), 2)
        ok_jpg, buf = cv2.imencode(".jpg", disp)
        if ok_jpg:
            with _frame_lock:
                _last_jpeg = buf.tobytes()

    cap.release()
    ser.close()
    print("[WORKER] stopped")


# ================== API ==================
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
        "rearm_delay_s": REARM_DELAY_S,
        "demo_cards_only": DEMO_CARDS_ONLY,
        "send_enabled": (_can_send_for(_current_model_code) if _current_model_code else None),
        "loaded": True if (_current_model_code) else False,
    }

@app.post("/v1/load_model")
def load_model(req: LoadModelReq):
    global _stop_flag, _current_thread, _current_model_code

    model_code: Optional[str] = None
    if req.name and req.name in DISPLAY_TO_CODE:
        model_code = DISPLAY_TO_CODE[req.name]
    elif req.model:
        code = MODEL_ALIASES.get(req.model, req.model)
        if code in MODEL_PATHS:
            model_code = code

    if not model_code:
        return {"ok": False, "error": f"unknown model: name={req.name}, model={req.model}"}

    if _current_thread and _current_thread.is_alive():
        _stop_flag = True
        _current_thread.join()

    _stop_flag = False
    _current_model_code = model_code
    send_enabled = _can_send_for(model_code)
    _current_thread = Thread(target=yolo_worker, args=(model_code, send_enabled), daemon=True)
    _current_thread.start()
    return {"ok": True, "model": CODE_TO_DISPLAY.get(model_code, model_code), "loaded": True}

@app.post("/v1/stop")
def stop():
    global _stop_flag, _current_thread
    _stop_flag = True
    if _current_thread:
        _current_thread.join()
    _current_thread = None
    return {"ok": True}

# (기존 predict 유지 — 필요시 클라이언트 캡쳐 방식)
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
    with _predict_lock:
        res = m(img)
    r = res[0]
    preds: List[Dict[str, Any]] = []

    if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
        names = getattr(r, 'names', None) or getattr(m, 'names', {})
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0]) if hasattr(b, 'conf') else None
            idx  = int(b.cls[0])   if hasattr(b, 'cls') else None
            label = names.get(idx, str(idx)) if isinstance(names, dict) else str(idx)
            preds.append({"label": label, "conf": conf, "box": [x1, y1, max(0, x2-x1), max(0, y2-y1)]})
    else:
        probs = getattr(r, 'probs', None)
        if probs is not None:
            k = int(probs.top1)
            conf = float(getattr(probs, 'top1conf', probs.data[k]))
            names = getattr(r, 'names', None) or getattr(m, 'names', {})
            label = names.get(k, str(k)) if isinstance(names, dict) else str(k)
            h, w = img.shape[:2]
            preds = [{"label": label, "conf": conf, "box": [int(w*0.3), int(h*0.3), int(w*0.4), int(h*0.4)]}]

    return {"ok": True, "model": _current_model_code, "preds": preds}

# ★ MJPEG 스트림
@app.get("/v1/stream")
async def stream():
    boundary = "frame"

    async def gen():
        while True:
            with _frame_lock:
                data = _last_jpeg
            if data is None:
                # 초기 대기화면
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for camera/worker...", (18, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
                _, b = cv2.imencode(".jpg", blank)
                data = b.tobytes()

            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" +
                data + b"\r\n"
            )
            await asyncio.sleep(0.07)   # ~14fps

    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

# 정적 파일
app.mount("/", StaticFiles(directory=".", html=True), name="static")
