##모델 지정 테스트용
# classify_vote_3s_send_once.py
# pip install ultralytics opencv-python pyserial
import os, time, cv2, serial
import numpy as np
from ultralytics import YOLO
# ======== 모델 선택 & 경로 (여기만 바꾸면 됩니다) ========
# MODEL_CODE: "beef" | "beverage" | "recycle" | "playing_cards"
MODEL_CODE = "beverage"
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
# 라벨 → 패킷 문자
def get_packet_char(model_name, label):
    if model_name == "playing_cards":
        return label[-1].lower()  # 끝문자 C/D/H/S → c/d/h/s
    mapping = {
        "beef":     {'grade_1':'s','grade_1p':'c','grade_1pp':'h','grade_2':'d'},
        "beverage": {'CocaColaKorea':'s','DongwonFB':'c','Haitaihtb':'h','LotteChilsung':'d'},
        "recycle":  {'plastic':'s','glass':'c','metal':'h','cardboard':'d'},
    }
    return mapping.get(model_name, {}).get(label, None)
# ======== 투표/쿨다운 설정 ========
VOTING_WINDOW_S = 3.0    # 3초 투표
START_THRES     = 0.15   # 투표 시작 문턱(낮게/0도 가능)
REARM_DELAY_S   = 1.0    # 전송 후 재무장 대기(중복 방지)
# ======== 시리얼/카메라/ROI ========
SERIAL_PORT = os.environ.get("SERIAL_PORT", "/dev/tty.usbserial-0001")
BAUDRATE    = 115200
CAM_INDEX   = 0
FRAME_W, FRAME_H = 720, 480
USE_ROI = True
ROI_L, ROI_R = 0.33, 1.0 - 4.0/9.0  # 가운데 대역
DEVICE = "mps"  # 맥이면 mps, 아니면 자동 cpu
def apply_roi(img):
    if not USE_ROI:
        return img
    h, w = img.shape[:2]
    left  = int(w * ROI_L)
    right = int(w * ROI_R)
    masked = np.zeros_like(img)
    masked[:, left:right, :] = img[:, left:right, :]
    return masked
def main():
    model_path = MODEL_PATHS[MODEL_CODE]
    classnames = MODEL_CLASSNAMES[MODEL_CODE]
    model = YOLO(model_path)
    # 시리얼 & 카메라
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    vote_active = False
    vote_start  = 0.0
    votes_vec   = None
    next_rearm_time = 0.0
    print(f"[START] {MODEL_CODE} (3s voting, send once). Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = apply_roi(frame)
        t_now = time.perf_counter()
        results = model.predict(img, device=DEVICE, verbose=False)
        r = results[0]
        disp = img.copy()
        if hasattr(r, "probs") and (r.probs is not None):
            probs = r.probs.data
            probs = probs.cpu().numpy() if hasattr(probs, "cpu") else np.asarray(probs)
            k = int(np.argmax(probs))
            p = float(probs[k])
            # 투표 시작(쿨다운 통과)
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
                    pkt = get_packet_char(MODEL_CODE, final_label)
                    if pkt:
                        ser.write(pkt.encode())   # ★ 한 글자만 전송(개행 없음)
                        print(f"[SEND] {final_label} -> '{pkt}'")
                    else:
                        print(f"[SEND] skipped (no pkt for {final_label})")
                    # 리셋 + 쿨다운
                    vote_active = False
                    votes_vec   = None
                    next_rearm_time = time.perf_counter() + REARM_DELAY_S
            # 오버레이
            cv2.putText(disp, f"top1: {classnames[k]} ({p:.2f})",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(disp, f"vote:{'ON' if vote_active else 'OFF'}  cd:{max(0.0, next_rearm_time - t_now):.1f}s",
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)
        cv2.imshow("classify-3s-vote-send-once", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
    print("[END]")
if __name__ == "__main__":
    main()
