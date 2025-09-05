# -*- coding: utf-8 -*-
import base64, io, json, os, threading, time
from typing import Any, Dict, List, Optional
from flask import Flask, request, Response, render_template
from PIL import Image
import numpy as np

YOLO_AVAILABLE = True
try:
    from ultralytics import YOLO
except Exception:
    YOLO_AVAILABLE = False

# ── 변경 ①: 정적/템플릿을 현재 폴더에서 직접 서빙 ( /static 접두사 없이 logo.png 등 접근 )
app = Flask(
    __name__,
    template_folder=".",   # index.html이 현재 폴더에 있을 때
    static_folder=".",     # 이미지/기타 정적 파일도 현재 폴더
    static_url_path=""     # /logo.png 처럼 바로 접근
)

# 사용 중인 경로(스크린샷 기준)
MODEL_REGISTRY = {
    "cards":    "/Users/abitria/coding/python/yolov8s_playing_cards.pt",
    "beef":     "/Users/abitria/coding/python/beef.pt",
    "beverage": "/Users/abitria/coding/python/beverage.pt",
    "recycle":  "/Users/abitria/coding/python/recycle.pt",
}

# (선택) 한글 표기 → 내부 ID 매핑도 받아주기 위함
KOR_TO_ID = {
    "트럼프 카드": "cards",
    "고기등급": "beef",
    "음료수 종류": "beverage",
    "재활용품 분류": "recycle",
}

current_model_id: Optional[str] = None
current_model = None
model_lock = threading.Lock()

def _load_model(model_id: str):
    """model_id: cards | beef | beverage | recycle"""
    global current_model, current_model_id
    if not YOLO_AVAILABLE:
        current_model = None
        current_model_id = model_id
        return

    weights = MODEL_REGISTRY.get(model_id)
    if not weights or not os.path.exists(weights):
        raise FileNotFoundError(f"weights not found: {weights}")

    try:
        m = YOLO(weights)
    except Exception as e:
        # Ultralytics 체크포인트가 아닐 때 흔한 에러(KeyError('model')) 등을 명확히 안내
        raise RuntimeError(
            f"failed to load weights: {weights}. "
            f"Ultralytics YOLOv8 형식의 best.pt인지 확인하세요. 원인: {e}"
        )

    current_model = m
    current_model_id = model_id

# 서버 시작 시, 존재하는 첫 모델 자동 로드
for _mid, _mpath in MODEL_REGISTRY.items():
    if os.path.exists(_mpath):
        try:
            _load_model(_mid)
            break
        except Exception:
            pass

def _decode_data_url_to_ndarray(data_url: str) -> np.ndarray:
    if not data_url.startswith("data:"):
        raise ValueError("invalid data url")
    content = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(content)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)

@app.get("/")
def index():
    # 같은 폴더의 index.html 표시
    return render_template("index.html")

def _coerce_model_id(payload: Dict[str, Any]) -> Optional[str]:
    """
    payload에 {model:'beef'} 또는 {name:'음료수 종류'} 둘 다 허용
    """
    mid = (payload.get("model") or "").strip()
    if mid in MODEL_REGISTRY:
        return mid
    name = (payload.get("name") or "").strip()
    if name and name in KOR_TO_ID:
        return KOR_TO_ID[name]
    return None

@app.post("/v1/switch_model")
def switch_model():
    data = request.get_json(force=True, silent=True) or {}
    model_id = _coerce_model_id(data)
    if not model_id:
        return Response(json.dumps({"ok": False, "error": "missing or unknown model"}), mimetype="application/json")
    with model_lock:
        try:
            _load_model(model_id)
            return Response(json.dumps({"ok": True, "model": model_id, "loaded": current_model is not None}), mimetype="application/json")
        except Exception as e:
            return Response(json.dumps({"ok": False, "model": model_id, "error": str(e)}), mimetype="application/json")

# ── 변경 ②: 프런트가 /v1/load_model 로 호출해도 동작하도록 별칭 추가
@app.post("/v1/load_model")
def load_model_alias():
    return switch_model()

@app.get("/v1/status")
def status():
    return Response(json.dumps({
        "ok": True,
        "model": current_model_id,
        "loaded": current_model is not None
    }), mimetype="application/json")

@app.post("/v1/predict")
def predict():
    t0 = time.time()
    data = request.get_json(force=True, silent=True) or {}
    img64 = data.get("image")
    if not img64:
        return Response(json.dumps({"ok": False, "error": "missing image"}), mimetype="application/json")
    try:
        img = _decode_data_url_to_ndarray(img64)
    except Exception as e:
        return Response(json.dumps({"ok": False, "error": f"decode fail: {e}"}), mimetype="application/json")

    preds: List[Dict[str, Any]] = []
    with model_lock:
        m = current_model
        mid = current_model_id

    if m is None or not YOLO_AVAILABLE:
        h, w = img.shape[:2]
        preds = [{"label": mid or "dummy", "conf": 0.0, "box": [int(w*0.3), int(h*0.3), int(w*0.4), int(h*0.4)]}]
    else:
        try:
            # 분류/감지 모두 지원 (Ultralytics는 자동으로 task 인식)
            res = m(img)
            r = res[0]
            names = getattr(r, 'names', None) or getattr(m, 'names', {})
            # detect
            if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    xyxy = b.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    bw = max(0, x2 - x1); bh = max(0, y2 - y1)
                    conf = float(b.conf[0]) if hasattr(b, 'conf') else None
                    cls_idx = int(b.cls[0]) if hasattr(b, 'cls') else None
                    label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
                    preds.append({"label": label, "conf": conf, "box": [x1, y1, bw, bh]})
            else:
                # classify
                probs = getattr(r, 'probs', None)
                if probs is not None and hasattr(probs, 'top1'):
                    cls_idx = int(probs.top1)
                    conf = float(getattr(probs, 'top1conf', None) or probs.data[cls_idx].item())
                    label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
                    h, w = img.shape[:2]
                    preds.append({"label": label, "conf": conf, "box": [int(w*0.3), int(h*0.3), int(w*0.4), int(h*0.4)]})
        except Exception as e:
            return Response(json.dumps({"ok": False, "error": f"inference error: {e}"}), mimetype="application/json")

    dt = int((time.time() - t0) * 1000)
    return Response(json.dumps({"ok": True, "model": current_model_id, "preds": preds, "time_ms": dt}), mimetype="application/json")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
