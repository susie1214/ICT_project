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
MODEL_CODE = "recycle"

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
SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/cu.usbserial-0001")
BAUDRATE    = 115200
CAM_INDEX   = 0
FRAME_W, FRAME_H = 720, 480
USE_ROI = True
ROI_L, ROI_R = 0.33, 1.0 - 4.0/9.0
DEVICE = "mps"  # Mac(MPS). 에러나면 "cpu"

# ===== 초록 배경 무시 옵션 =====
IGNORE_GREEN   = True         # 초록(벨트/배경) 제거 ON/OFF
PROTECT_CENTER = True         # 중앙 통로(카드 지나는 구역)는 보호
AUTO_GREEN     = True         # 시작 시 초록 톤 자동 캘리브레이션
GREEN_LOWER    = np.array([35, 40, 40],  dtype=np.uint8)  # 초기값(현장에 맞게 조정 가능)
GREEN_UPPER    = np.array([85, 255, 255], dtype=np.uint8)
GREEN_BOX_MAX_RATIO = 0.60    # 박스 내부가 초록으로 60% 이상이면 무시
_auto_green_done = False

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
        return label[-1].lower() if label else None
    if model_name == "beverage":
        if _bev_map_fn:
            return _bev_map_fn(label)
        # fallback (필요 시 라벨명 맞춰 수정)
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

# ===== 초록 마스크/제거 =====
_kernel5 = np.ones((5,5), np.uint8)

def compute_green_mask(img_bgr):
    """현재 GREEN_LOWER/UPPER를 이용해 초록 마스크 생성(+노이즈 정리, 중앙 보호)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _kernel5, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel5, 1)
    if PROTECT_CENTER:
        h, w = img_bgr.shape[:2]
        cx0, cx1 = int(w*0.30), int(w*0.70)
        cy0, cy1 = int(h*0.15), int(h*0.85)
        mask[cy0:cy1, cx0:cx1] = 0
    return mask

def remove_green_background(img_bgr):
    """초록 벨트/배경을 회색으로 덮어 모델이 못 보게 함."""
    if not IGNORE_GREEN:
        return img_bgr
    mask = compute_green_mask(img_bgr)
    out = img_bgr.copy()
    out[mask > 0] = (128, 128, 128)   # 무채색으로 덮기
    return out

def autocalib_green_from_edges(frame_bgr):
    """프레임 상/하단 띠 샘플링으로 초록 HSV 범위를 추정."""
    h, w = frame_bgr.shape[:2]
    band = np.vstack([frame_bgr[0:int(0.15*h), :], frame_bgr[int(0.85*h):, :]])
    hsv  = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    h_med = np.median(hsv[:,:,0]); s_med = np.median(hsv[:,:,1])
    lo = np.array([max(0,   int(h_med-10)), max(0, int(s_med-40)), 30], dtype=np.uint8)
    hi = np.array([min(179, int(h_med+10)), 255,                     255], dtype=np.uint8)
    return lo, hi

# ===== 아두이노 READY 동기화 =====
def drain_logs(ser):
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
    global _auto_green_done, GREEN_LOWER, GREEN_UPPER

    path = MODEL_PATHS[MODEL_CODE]
    model = YOLO(path)
    is_classify = (getattr(model, "task", None) == "classify")
    print(f"[INFO] model.task={getattr(model,'task',None)}  SERIAL_PORT={SERIAL_PORT}")

    # 시리얼
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
    time.sleep(0.2)
    drain_logs(ser)

    # LCD 모델 전환(1=CARD, 2=RECYCLE, 3=BEVERAGE, 4=BEEF)
    select_map = {'playing_cards':'1', 'recycle':'2', 'beverage':'3', 'beef':'4'}
    sel = select_map.get(MODEL_CODE)
    if sel:
        print(f"[SER] send model select '{sel}' + reset count 'R'")
        ser.write(sel.encode()); time.sleep(0.05)
        ser.write(b'R');         time.sleep(0.05)
        drain_logs(ser)

    # 카메라
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
        if not ok:
            break

        # 0) (선택) 초록 자동 캘리브레이션: 시작 1회
        if AUTO_GREEN and not _auto_green_done:
            GREEN_LOWER, GREEN_UPPER = autocalib_green_from_edges(frame)
            _auto_green_done = True
            print(f"[GREEN] AUTO calib LOWER={GREEN_LOWER.tolist()} UPPER={GREEN_UPPER.tolist()}")

        # 1) ROI → 2) 초록 제거
        roi_img = apply_roi(frame)
        gmask = compute_green_mask(roi_img) if IGNORE_GREEN else None
        img   = remove_green_background(roi_img)

        tnow = time.perf_counter()

        # 3) 추론
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
            # detect: 박스 conf 합산 (초록 비율 높은 박스는 제외)
            n_classes = len(names) if names else (getattr(getattr(model,"model",None),"nc",0) or 0)
            frame_scores = np.zeros(n_classes, dtype=np.float32)
            if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                h, w = img.shape[:2]
                for b in res.boxes:
                    cls = int(b.cls[0]); conf = float(b.conf[0]) if hasattr(b,"conf") else 1.0
                    if not (0 <= cls < n_classes):
                        continue
                    if gmask is not None:
                        x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                        x1 = max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
                        if x2<=x1 or y2<=y1:
                            continue
                        crop = gmask[y1:y2, x1:x2]
                        green_ratio = float((crop>0).mean()) if crop.size else 0.0
                        if green_ratio >= GREEN_BOX_MAX_RATIO:
                            continue  # 초록영역 위 박스 → 무시
                    frame_scores[cls] += conf
                    start_ok = True
                k = int(np.argmax(frame_scores)); p = frame_scores[k]
                hint = f"det: {names[k] if k < len(names) else k} (+{p:.2f})"
            else:
                hint = "No boxes"
                frame_scores = np.zeros(len(names), dtype=np.float32)

        # 4) 투표 상태머신
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
                    print("[SER] wait READY…")
                    if wait_ready(ser, timeout=10.0):
                        ser.write(pkt.encode())   # 한 글자만 전송
                        print(f"[SEND] {final_label} -> '{pkt}'")
                    else:
                        print("[WARN] READY 미수신 → 전송 생략(유실 방지)")
                else:
                    print(f"[SEND] skipped (no pkt for '{final_label}')")

                vote_active = False
                votes_vec   = None
                next_rearm  = time.perf_counter() + REARM_DELAY_S

        # 5) 화면 오버레이
        disp = img.copy()
        cv2.putText(disp, hint, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(30,255,255),2)
        cv2.putText(disp, f"vote:{'ON' if vote_active else 'OFF'}  cd:{max(0.0,next_rearm-tnow):.1f}s",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(200,255,200),2)

        drain_logs(ser)  # MCU 로그 콘솔 출력
        cv2.imshow("classify/detect-3s-vote (green-masked)", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows(); ser.close()
    print("[END]")

if __name__ == "__main__":
    main()
