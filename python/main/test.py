# classify_or_detect_vote_3s_send_once.py
# pip install ultralytics opencv-python pyserial

import os, time, cv2, serial
import numpy as np
from ultralytics import YOLO
from label_maps import beverage_label_to_packet  

# ===== 모델 선택 (이 한 줄만 바꿔 테스트) =====
MODEL_CODE = "playing_cards"   # beef | beverage | recycle | playing_cards


def get_packet_char(model_name, label):
    if model_name == "playing_cards":
        return label[-1].lower()

    if model_name == "beverage":
        return beverage_label_to_packet(label)

    # 나머지 모델(고기/재활용)은 기존 매핑 유지
    mapping = {
        "beef":     {'grade_1':'s','grade_1p':'c','grade_1pp':'h','grade_2':'d'},
        "recycle":  {'plastic':'s','glass':'c','metal':'h','cardboard':'d'},
    }
    for_map = mapping.get(model_name, {})
    return for_map.get(label)


# 경로
MODEL_PATHS = {
    "playing_cards": "/Users/abitria/coding/python/yolov8s_playing_cards.pt",
    "beef":          "/Users/abitria/coding/python/beef.pt",
    "beverage":      "/Users/abitria/coding/python/beverage.pt",
    "recycle":       "/Users/abitria/coding/python/recycle.pt",
}



# ===== 투표/쿨다운 =====
VOTING_WINDOW_S = 2.5
START_THRES     = 0.15
REARM_DELAY_S   = 4.0

# ===== 장치/ROI =====
SERIAL_PORT = "/dev/cu.usbserial-0001"   # 포트 맞게 조정
BAUDRATE    = 115200
CAM_INDEX   = 0
FRAME_W, FRAME_H = 720, 480
USE_ROI = True
ROI_L, ROI_R = 0.33, 1.0 - 4.0/9.0
DEVICE = "mps"  # Mac

def apply_roi(img):
    if not USE_ROI: return img
    h, w = img.shape[:2]
    L, R = int(w*ROI_L), int(w*ROI_R)
    m = np.zeros_like(img); m[:, L:R, :] = img[:, L:R, :]
    return m

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

def main():
    path = MODEL_PATHS[MODEL_CODE]
    model = YOLO(path)
    is_classify = (getattr(model, "task", None) == "classify")
    print(f"[INFO] model.task={getattr(model,'task',None)}")

    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    vote_active = False
    vote_start  = 0.0
    votes_vec   = None
    next_rearm  = 0.0
    names       = None  # 모델에서 동적 획득

    print("[START] 3s voting / send once. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok: break
        img  = apply_roi(frame)
        tnow = time.perf_counter()

        res = model.predict(img, device=DEVICE, verbose=False)[0]
        if names is None:
            names = names_from_model(model, res)
            print(f"[INFO] class names = {names} (n={len(names)})")

        hint = ""
        frame_scores = None
        start_ok = False

        if is_classify and getattr(res, "probs", None) is not None:
            probs = res.probs.data
            probs = probs.cpu().numpy() if hasattr(probs, "cpu") else np.asarray(probs)
            # 길이 불일치 방어
            if probs.shape[-1] != len(names):
                print(f"[WARN] class count mismatch: probs={probs.shape[-1]} vs names={len(names)} → refresh names from model/result")
                names = names_from_model(model, res)
                print(f"[INFO] new names = {names} (n={len(names)})")
            frame_scores = probs.astype(np.float32)  # (nc,)
            k = int(np.argmax(probs)); p = float(probs[k])
            hint = f"top1: {names[k] if k < len(names) else k} ({p:.2f})"
            start_ok = (p >= START_THRES)

        else:
            # detect: 박스들의 conf로 누적
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

        # 투표 상태머신
        if (not vote_active) and (tnow >= next_rearm) and start_ok:
            vote_active = True
            vote_start  = tnow
            votes_vec   = np.zeros_like(frame_scores, dtype=np.float32)
            print("[VOTE] start (3s)")

        if vote_active:
            # 만약 프레임마다 길이가 변하면(드물지만) 맞춰줌
            if votes_vec.shape != frame_scores.shape:
                print(f"[WARN] score shape changed {votes_vec.shape} -> {frame_scores.shape}, reinit")
                votes_vec = np.zeros_like(frame_scores, dtype=np.float32)
            votes_vec += frame_scores
            if (tnow - vote_start) >= VOTING_WINDOW_S:
                final_k = int(np.argmax(votes_vec))
                final_label = names[final_k] if final_k < len(names) else str(final_k)
                pkt = get_packet_char(MODEL_CODE, final_label)
                if pkt:
                    ser.write(pkt.encode())
                    print(f"[SEND] {final_label} -> '{pkt}'")
                else:
                    print(f"[SEND] skipped (no pkt for '{final_label}')  <- 매핑 확인 필요")
                vote_active = False
                votes_vec   = None
                next_rearm  = time.perf_counter() + REARM_DELAY_S

        # 오버레이
        disp = img.copy()
        cv2.putText(disp, hint, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(30,255,255),2)
        cv2.putText(disp, f"vote:{'ON' if vote_active else 'OFF'}  cd:{max(0.0,next_rearm-tnow):.1f}s",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(200,255,200),2)

        cv2.imshow("classify/detect-3s-vote", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows(); ser.close()
    print("[END]")

if __name__ == "__main__":
    main()
