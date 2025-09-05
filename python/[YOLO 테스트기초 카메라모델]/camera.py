import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, c = frame.shape
    # 중앙 1/3의 좌우 범위 계산
    left = w // 3
    right = w - int(w * 3.5) // 9
    # 검정 마스크 생성
    mask = np.zeros_like(frame)
    # 중앙 1/3 영역만 원본 값 복사
    mask[:, left:right, :] = frame[:, left:right, :]
    cv2.imshow("Original", frame)
    cv2.imshow("Center 1/3 Only", mask)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()