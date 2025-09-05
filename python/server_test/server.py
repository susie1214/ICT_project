import os
import time
import threading
from threading import Lock
from typing import Dict, List, Optional
import cv2
import numpy as np
import serial
from ultralytics import YOLO
import cvzone
from flask import Flask, request, jsonify
# ========================
# [A] 분류모델 클래스/레이블 정의
# ========================
MODEL_CLASSNAMES: Dict[str, List[str]] = {
    "playing_cards": [
        '10C','10D','10H','10S','2C','2D','2H','2S',
        '3C','3D','3H','3S','4C','4D','4H','4S',
        '5C','5D','5H','5S','6C','6D','6H','6S',
        '7C','7D','7H','7S','8C','8D','8H','8S',
        '9C','9D','9H','9S','AC','AD','AH','AS',
        'JC','JD','JH','JS','KC','KD','KH','KS',
        'QC','QD','QH','QS'
    ],
    "beef": ['grade_1', 'grade_1p', 'grade_1pp', 'grade_2', 'grade_3'],
    "beverage": ['CocaColaKorea', 'DongwonFB','Haitaihtb','LotteChilsung','WoongjinFoods'],
    "recycle": ['plastic', 'glass', 'metal', 'cardboard', 'battery'],
}
def card_packet(label: str) -> Optional[str]:
    return label[-1].lower() if label else None
LABEL_TO_PACKET = {
    "playing_cards": card_packet,
    "beef": {
        'grade_1':  'h',
        'grade_1p': 'c',
        'grade_1pp':'s',
        'grade_2':  'd',
        # 'grade_3': 패스(무시)
    },
    "beverage": {
        'CocaColaKorea': 's',
        'DongwonFB':     'c',
        'Haitaihtb':     'h',
        'LotteChilsung': 'd',
        # 'WoongjinFoods': 패스
    },
    "recycle": {
        'plastic':  's',
        'glass':    'c',
        'metal':    'h',
        'cardboard':'d',
        # 'battery':  패스
    },
}
# ==============================
# [B] pt모델 파일 경로 (파일명 꼭 확인!)
# ==============================
MODEL_PATHS: Dict[str, str] = {
    "playing_cards": "/Users/abitria/coding/python/yolo8s_playing_cards.pt",  # 실제 경로와 파일명 맞추기!
    "beef":          "/Users/abitria/coding/python/beef.pt",
    "beverage":      "/Users/abitria/coding/python/beverage.pt",
    "recycle":       "/Users/abitria/coding/python/recycle.pt",
}
# ==============================
# [C] 하드웨어/운영 옵션
# ==============================
DEVICE = os.environ.get("DEVICE", None)
CAM_INDEX = int(os.environ.get("CAM_INDEX", "0"))
SERIAL_PORT = os.environ.get("SERIAL_PORT", "/dev/tty.usbserial-0001")
BAUD = int(os.environ.get("BAUD", "115200"))
CONF_TH = float(os.environ.get("CONF_TH", "0.5"))
SEND_COOLDOWN = float(os.environ.get("SEND_COOLDOWN", "0.7"))
SHOW_WINDOW = os.environ.get("SHOW_WINDOW", "1") == "1"
def get_packet_char(model_name: str, label: str) -> Optional[str]:
    mapper = LABEL_TO_PACKET.get(model_name)
    if mapper is None:
        return None
    if callable(mapper):
        return mapper(label)
    return mapper.get(label, None)
# ==============================
# [D] YOLO 엔진/시리얼 워커 클래스
# ==============================
class YoloWorker:
    def __init__(self):
        self._lock = Lock()
        self._running = True
        self._current_model_name = None
        self._model: Optional[YOLO] = None
        self._class_names: List[str] = []
        self._cap = None
        self._ser = None
        self._last_send_ts = 0.0
    def open_devices(self):
        self._cap = cv2.VideoCapture(CAM_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  720)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
        print(f"[SERIAL] Opened: {SERIAL_PORT} @ {BAUD}")
    def close_devices(self):
        if self._cap:
            self._cap.release()
        if self._ser:
            self._ser.close()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
    def load_model(self, model_name: str) -> Dict:
        if model_name not in MODEL_PATHS:
            return {"ok": False, "message": f"Unknown model: {model_name}"}
        model_path = MODEL_PATHS[model_name]
        if not os.path.exists(model_path):
            return {"ok": False, "message": f"Model file not found: {model_path}"}
        with self._lock:
            print(f"[YOLO] Loading {model_name}: {model_path}")
            self._model = YOLO(model_path)
            self._class_names = MODEL_CLASSNAMES[model_name]
            self._current_model_name = model_name
        return {"ok": True, "message": f"Loaded: {model_name}"}
    def set_running(self, flag: bool) -> None:
        with self._lock:
            self._running = flag
    def run_forever(self):
        self.open_devices()
        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    print("[WARN] Camera read fail")
                    time.sleep(0.1)
                    continue
                with self._lock:
                    running = self._running
                    model = self._model
                    model_name = self._current_model_name
                    class_names = self._class_names
                if not running or model is None:
                    if SHOW_WINDOW:
                        cv2.imshow("YOLO (idle: no model/paused)", frame)
                        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                            break
                    time.sleep(0.03)
                    continue
                h, w, _ = frame.shape
                left = w // 3
                right = w - w * 4 // 9
                roi = np.zeros_like(frame)
                roi[:, left:right, :] = frame[:, left:right, :]
                results = model(roi, conf=CONF_TH, verbose=False, device=DEVICE)
                sent_this_frame = False
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = class_names[cls]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        pkt = get_packet_char(model_name, label)
                        color = (0, 255, 0) if pkt else (128, 128, 128)
                        now = time.time()
                        if (conf >= CONF_TH) and pkt and not sent_this_frame and (now - self._last_send_ts >= SEND_COOLDOWN):
                            print(f"[SEND] {label} ({conf:.2f}) -> '{pkt}'")
                            self._ser.write(pkt.encode())
                            self._last_send_ts = now
                            sent_this_frame = True
                        elif conf >= CONF_TH and not pkt:
                            print(f"[PASS] {label} ({conf:.2f})")
                        if SHOW_WINDOW:
                            cvzone.cornerRect(roi, (x1, y1, x2 - x1, y2 - y1), l=9, t=2, colorC=color)
                            cvzone.putTextRect(roi, f"{label} {conf:.2f}", (x1, max(0, y1 - 10)),
                                               scale=1, thickness=2, colorT=color)
                if SHOW_WINDOW:
                    vis = frame.copy()
                    vis[:, left:right, :] = roi[:, left:right, :]
                    title = f"YOLO - {model_name}" if model_name else "YOLO"
                    cv2.imshow(title, vis)
                    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                        break
        finally:
            self.close_devices()
# ==============================
# [E] Flask HTTP 서버(앱/웹에서 REST 제어)
# ==============================
app = Flask(__name__)
WORKER = YoloWorker()
@app.get("/v1/status")
def http_status():
    return jsonify({"ok": True, "current_model": WORKER._current_model_name})
@app.post("/v1/switch_model")
def http_switch_model():
    data = request.get_json(force=True, silent=True) or {}
    model = (data.get("model") or "").strip()
    if not model:
        return jsonify({"ok": False, "message": "model is required"}), 400
    res = WORKER.load_model(model)
    return jsonify(res), (200 if res.get("ok") else 400)
@app.post("/v1/stop")
def http_stop():
    WORKER.set_running(False)
    return jsonify({"ok": True, "message": "paused"})
@app.post("/v1/start")
def http_start():
    WORKER.set_running(True)
    return jsonify({"ok": True, "message": "running"})
if __name__ == "__main__":
    th = threading.Thread(target=WORKER.run_forever, daemon=True)
    th.start()
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)