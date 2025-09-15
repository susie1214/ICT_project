# Delta Robot Project

---

실시간 비전 인식(YOLOv8-s) + 음성인식(Vosk STT) + 델타 로봇 제어를 하나로 통합한 교육/전시용 프로젝트입니다.  
웹 UI(마이크 버튼, 모델 전환, 프리뷰)에서 명령을 보내면, 백엔드가 인식 결과를 시리얼로 OpenRB-150/AX-12 로봇에 전달합니다.

---
<!--<p align="center"> <img src="docs/img/robot-conveyor.png" alt="Robot & Conveyor" width="600"> </p>-->

데모 대상: 트럼프 카드 / 재활용 / 음료 / 소고기 등급  
OS: Windows 10/11, WSL2, 또는 Linux  
언어: Python (백엔드/AI), C++/Arduino (펌웨어), HTML/CSS/JS (웹)  

---

✨ 기능 요약

객체 인식: YOLOv8 가중치(yolov8s_playing_cards.pt, recycle.pt, beverage.pt, beef.pt)로 실시간 분류/검출

음성 인식(STT): 로컬 Vosk 모델(오프라인) 또는 브라우저-STT 대체

로봇 제어: USB-TTL 시리얼 프로토콜로 OpenRB-150/AX-12 제어(픽/플레이스)

웹 UI: Dark-Blue 톤 대시보드, 모델 전환, 미리보기, 상태 표시

확장: Hailo/Jetson, 추가 클래스, 파인튜닝/학습 스크립트, 데이터 라벨맵

---

📦 저장소 구조

```
delta_robot/
├─ web/                     # 웹/백엔드 (Flask or FastAPI + static)
│  ├─ app.py                # (기존 app12_mod.py 개명 권장)
│  ├─ requirements.txt
│  ├─ static/               # JS/CSS/Images
│  └─ templates/
│     └─ index.html
├─ firmware/                # (기존 c++_arduino/*)
│  ├─ arduino_final/
│  ├─ serial_test/
│  ├─ lcd_test/
│  ├─ pick_test/
│  └─ main/
├─ python/                  # (실험/학습/테스트 코드)
│  ├─ main/
│  ├─ server_test/
│  ├─ lcd_test/
│  ├─ label_maps/
│  ├─ yolo_basics/          # [YOLO 테스트] 기초 카메라모델
│  └─ vosk_stt_test/
├─ models/                  # (대용량 .pt 파일은 깃에 올리지 않음)
│  ├─ yolov8s_playing_cards.pt
│  ├─ recycle.pt
│  ├─ beverage.pt
│  └─ beef.pt
├─ stt/                     # (대용량 zip/wav는 깃 제외)
│  ├─ vosk-model-en-us-0.42-gigaspeech/   # 압축 해제 후 폴더
│  ├─ vosk-model-small-en-us-0.15/
│  ├─ trump_cards/          # TTS 음성들(mp3)
│  └─ Text to Speech.wav
├─ assets/                  # 포스터/영상/로고 등
│  ├─ logo.png
│  ├─ delta-logo.png
│  ├─ robot-conveyor.png
│  └─ video/
├─ scripts/
│  └─ download_models.bat   # (선택) 외부에서 모델 받는 스크립트
├─ .gitignore
├─ Dockerfile
├─ docker-compose.yml
└─ README.md
```

---

🔧 설치 & 빠른 실행 (로컬)

1) Python 가상환경 & 패키지
``` CMD
cd web
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
# ultralytics/torch 필요 시
pip install "torch==2.3.1" "torchvision==0.18.1" --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python pyserial vosk pyaudio
```

2) 모델/리소스 배치
```
models/ 폴더에 .pt 가중치 4개를 넣습니다.

stt/ 폴더에 Vosk 모델 zip을 압축해제한 폴더를 둡니다. (예: stt/vosk-model-en-us-0.42-gigaspeech/)
```

3) 서버 실행
```
# 예: Flask 파일명이 app.py (기존 app12_mod.py와 동일 기능)
python web/app.py \
  --model models/yolov8s_playing_cards.pt \
  --stt_dir stt/vosk-model-en-us-0.42-gigaspeech \
  --serial COM5 \
  --baud 115200 \
  --port 8000
```

브라우저에서 http://localhost:8000 접속

웹캠 권한 허용

마이크 STT(브라우저/로컬 Vosk 중 하나) → 인식 결과 → YOLO 클래스 매칭 → 시리얼 전송

Windows 시리얼 포트: COMx / Linux: /dev/ttyUSB0 또는 /dev/ttyACM0

---

🧠 명령 & 클래스 매핑(예시)

“heart seven” → 7H 🂷

“spade number three” → 3S

재활용/음료/고기 모드 전환: UI의 모델 드롭다운 또는 /api/switch_model 호출

매칭 성공 시, 시리얼로 {mode}:{class}:{x}:{y} 전송(예시)

실제 매핑/프로토콜은 web/app.py의 테이블을 확인/수정하세요.

---

🧪 카메라/학습 테스트

python/yolo_basics/ : 카메라 기초 테스트

python/label_maps/ : 클래스 레이블 참고

학습/파인튜닝은 Ultralytics CLI 또는 스크립트로 별도 진행 권장

---

🧵 펌웨어(Arduino/OpenRB-150)

firmware/arduino_final/ : 실제 전시용 펌웨어

firmware/serial_test/ : 시리얼 수신→동작 테스트

firmware/pick_test/ : 흡착/픽업 테스트

프로토콜: MODE,CLASS,X,Y\n 혹은 단순 CLASS\n (펌웨어 코드와 동일하게 맞추기)

<img width="325" height="202" alt="image" src="https://github.com/user-attachments/assets/13a91de5-e2f5-4851-ac73-32f387a56596" />


