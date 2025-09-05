# classify_or_detect_vote_3s_send_once.py
# pip install ultralytics opencv-python pyserial

import os, time, cv2, serial
import numpy as np
from ultralytics import YOLO

# --- (선택) beverage 라벨 맵이 별도 파일이면 임포트 / 없으면 fallback 사용 ---
try:
    from label_maps import beverage_label_to_packet as _bev_map_fn  # optional
except Exception:
    _bev_map_fn = None

# ===== 모델 선택 (이 한 줄만 바꿔 테스트) =====
# "beef" | "beverage" | "recycle" | "playing_cards"
MODEL_CODE = "playing_cards"

# ===== 경로 =====
MODEL_PATHS = {
    "playing_cards": "/Users/abitria/coding/python/yolov8s_playing_cards.pt",
    "beef":          "/Users/abitria/coding/python/beef.pt",
    "beverage":      "/Users/abitria/coding/python/beverage.pt",
    "recycle":       "/Users/abitria/coding/python/recycle.pt",
}

# ===== 투표/쿨다운 =====
VOTING_WINDOW_S = 4.0
START_THRES     = 0.15
REARM_DELAY_S   = 5.0

# ===== 장치/ROI =====
SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/cu.usbserial-0001")  # 환경변수로도 변경 가능
BAUDRATE    = 115200
CAM_INDEX   = 0
FRAME_W, FRAME_H = 720, 480
USE_ROI = True
ROI_L, ROI_R = 0.33, 1.0 - 4.0/9.0

DEVICE = "mps"  # Mac(MPS). 다른 플랫폼이면 'cpu' 추천

# ===== 라벨 이름(클래스 목록) 안전 추출 =====
def names_from_model(model, result=None):
    srcs = []
    if result is not None:
        srcs.append(getattr(result, "names", None))
    srcs.append(getattr(model, "names", None))
    for n in srcs:
        if isinstance(n, dict):
            return [n[i] for i in sorted(n)]
        if isinstance(n, (list, tuple)):
            return list(n)
    nc = getattr(getattr(model, "model", None), "nc", 0) or 0
    return [str(i) for i in range(nc)]

# ===== 라벨 → 패킷 =====
def get_packet_char(model_name, label):
    if model_name == "playing_cards":
        # 끝문자 C/D/H/S → c/d/h/s
        return label[-1].lower() if label else None

    if model_name == "beverage":
        if _bev_map_fn:
            return _bev_map_fn(label)
        # fallback (필요 시 여기 수정)
        fallback = {'CocaColaKorea':'s','DongwonFB':'c','Haitaihtb':'h','LotteChilsung':'d'}
        return fallback.get(label)

    mapping = {
        "beef":    {'grade_1':'s','grade_1p':'c','grade_1pp':'h','grade_2':'d'},
        "recycle": {'plastic':'s','glass':'c','metal':'h','cardboard':'d'},
    }
    return mapping.get(model_name, {}).get(label)

# ===== ROI =====
def apply_roi(img):
    if not USE_ROI: return img
    h, w = img.shape[:2]
    L, R = int(w*ROI_L), int(w*ROI_R)
    m = np.zeros_like(img); m[:, L:R, :] = img[:, L:R, :]
    return m

# ===== 아두이노 READY 대기 & 로그 드레인 =====
def drain_logs(ser):
    # 시리얼 로그를 읽어 화면에 찍어줌
    try:
        n = ser.in_waiting
        if n:
            data = ser.read(n)
            if data:
                try:
                    print("[MCU]", data.decode(errors="ignore").strip())
                except Exception:
                    print("[MCU bytes]", data)
    except Exception:
        pass

def wait_ready(ser, timeout=10.0):
    """아두이노가 'READY'를 보낼 때까지 대기 (최대 timeout초)"""
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

# ===== 메인 =====
def main():
    path = MODEL_PATHS[MODEL_CODE]
    model = YOLO(path)
    is_classify = (getattr(model, "task", None) == "classify")
    print(f"[INFO] model.task={getattr(model,'task',None)}  SERIAL_PORT={SERIAL_PORT}")

    # 시리얼/카메라
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
    time.sleep(0.2)  # 포트 settle
    drain_logs(ser)

    # ★ 아두이노 LCD 모델 전환: 1=CARD, 2=RECYCLE, 3=BEVERAGE, 4=BEEF
    select_map = {'playing_cards':'1', 'recycle':'2', 'beverage':'3', 'beef':'4'}
    sel = select_map.get(MODEL_CODE)
    if sel:
        print(f"[SER] send model select '{sel}'")
        ser.write(sel.encode())
        time.sleep(0.05)
        ser.write(b'R')  # (선택) 현재 모델 카운트 리셋
        time.sleep(0.05)
        drain_logs(ser)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    vote_active = False
    vote_start  = 0.0
    votes_vec   = None
    next_rearm  = 0.0
    names       = None

    print("[START] voting & send-once. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok: break
        img  = apply_roi(frame)
        tnow = time.perf_counter()

        res = model.predict(img, device=DEVICE, verbose=False)[0]
        if names is None:
            names = names_from_model(model, res)
            print(f"[INFO] classes = {names} (n={len(names)})")

        hint = ""
        start_ok = False

        if is_classify and getattr(res, "probs", None) is not None:
            probs = res.probs.data
            probs = probs.cpu().numpy() if hasattr(probs, "cpu") else np.asarray(probs)
            if probs.shape[-1] != len(names):
                print(f"[WARN] class count mismatch: probs={probs.shape[-1]} vs names={len(names)} → refresh")
                names = names_from_model(model, res)
            frame_scores = probs.astype(np.float32)
            k = int(np.argmax(probs)); p = float(probs[k])
            hint = f"top1: {names[k] if k < len(names) else k} ({p:.2f})"
            start_ok = (p >= START_THRES)

        else:
            # detect: 박스 conf 합산
            n_classes = len(names) if names else (getattr(getattr(model,"model",None),"nc",0) or 0)
            frame_scores = np.zeros(n_classes, dtype=np.float32)
            if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    cls = int(b.cls[0]); conf = float(b.conf[0]) if hasattr(b,"conf") else 1.0
                    if 0 <= cls < n_classes:
                        frame_scores[cls] += conf
                        start_ok = True
                k = int(np.argmax(frame_scores)); p = frame_scores[k]
                hint = f"det: {names[k] if k < len(names) else k} (+{p:.2f})"
            else:
                hint = "No boxes"
                frame_scores = np.zeros(len(names), dtype=np.float32)

        # --- 투표 상태머신 ---
        if (not vote_active) and (tnow >= next_rearm) and start_ok:
            vote_active = True
            vote_start  = tnow
            votes_vec   = np.zeros_like(frame_scores, dtype=np.float32)
            print("[VOTE] start")

        if vote_active:
            if votes_vec.shape != frame_scores.shape:
                print(f"[WARN] score shape changed {votes_vec.shape} -> {frame_scores.shape}, reinit")
                votes_vec = np.zeros_like(frame_scores, dtype=np.float32)
            votes_vec += frame_scores
            if (tnow - vote_start) >= VOTING_WINDOW_S:
                final_k = int(np.argmax(votes_vec))
                final_label = names[final_k] if final_k < len(names) else str(final_k)
                pkt = get_packet_char(MODEL_CODE, final_label)

                if pkt:
                    ser.write(pkt.encode())   # READY 대기 없이 곧바로 보냄
                    print(f"[SEND] {final_label} -> '{pkt}'")
                else:
                    print(f"[SEND] skipped (no pkt for '{final_label}')")

                vote_active = False
                votes_vec   = None
                next_rearm  = time.perf_counter() + REARM_DELAY_S

        # --- 화면 오버레이 ---
        disp = img.copy()
        cv2.putText(disp, hint, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(30,255,255),2)
        cv2.putText(disp, f"vote:{'ON' if vote_active else 'OFF'}  cd:{max(0.0,next_rearm-tnow):.1f}s",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(200,255,200),2)

        drain_logs(ser)  # MCU 로그를 콘솔로 확인
        cv2.imshow("classify/detect-3s-vote", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows(); ser.close()
    print("[END]")

if __name__ == "__main__":
    main()
