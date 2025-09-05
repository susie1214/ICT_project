#include <Dynamixel2Arduino.h>

// 포트 설정
#define DXL_SERIAL   Serial1    // Dynamixel 제어용 (왼쪽 RS485 포트)
#define CMD_SERIAL   Serial     // MacBook과 USB-C 통신용 (Serial = USB CDC)
#define DEBUG_SERIAL Serial2    // 보조 디버그용 (가운데 점퍼선 TX/RX, 선택사항)

const int DXL_DIR_PIN = -1;     // OpenRB-150은 자동 방향 제어
const uint8_t motor_ids[] = {1, 2, 3};
const uint8_t num_motors = 3;

#define RELAY_PIN 8             // 릴레이 흡착기 제어 핀

// 위치 정의 (기준에 따라 수정 가능)
const uint16_t pose_default[] = {310, 314, 337};
const uint16_t pose_pickup[]  = {353, 362, 369};
const uint16_t pose_spade[]   = {326, 435, 355};
const uint16_t pose_club[]    = {414, 358, 310};
const uint16_t pose_heart[]   = {289, 352, 424};
const uint16_t pose_dia[]     = {379, 269, 406};

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

// ── 부드러운 이동 함수 ──
void smoothMoveToPose(
  const uint16_t *target,
  const char* name,
  uint8_t steps = 20,
  uint16_t delayMs = 5
) {
  CMD_SERIAL.print("이동: "); CMD_SERIAL.println(name);

  uint16_t curr[num_motors];
  for (int i = 0; i < num_motors; i++) {
    curr[i] = dxl.getPresentPosition(motor_ids[i]);
  }

  for (int s = 1; s <= steps; s++) {
    float t = (float)s / steps;
    float ease = (1 - cos(t * PI)) / 2;

    for (int i = 0; i < num_motors; i++) {
      uint16_t pos = curr[i] + (target[i] - curr[i]) * ease;
      dxl.setGoalPosition(motor_ids[i], pos);
    }
    delay(delayMs);
  }
}

// 흡착기 제어
void suctionOn()  { digitalWrite(RELAY_PIN, HIGH); }
void suctionOff() { digitalWrite(RELAY_PIN, LOW);  }

// 픽업 및 배치 시퀀스
void pickupAndPlace(const uint16_t *dest, const char* label) {
  smoothMoveToPose(pose_default, "기본");   delay(200);
  smoothMoveToPose(pose_pickup,  "픽업");   delay(200);
  suctionOn(); delay(700);
  smoothMoveToPose(pose_default, "복귀");   delay(200);
  smoothMoveToPose(dest,         label);    delay(200);
  suctionOff(); delay(700);
  smoothMoveToPose(pose_default, "기본복귀");
}

void setup() {
  CMD_SERIAL.begin(115200);     // MacBook USB 시리얼 연결
  DEBUG_SERIAL.begin(115200);   // (선택) 보조 디버깅 UART

  dxl.begin(115200);
  dxl.setPortProtocolVersion(1.0);

  for (int i = 0; i < num_motors; i++) {
    dxl.torqueOff(motor_ids[i]);
    dxl.setOperatingMode(motor_ids[i], OP_POSITION);
    dxl.torqueOn(motor_ids[i]);
  }

  pinMode(RELAY_PIN, OUTPUT);
  suctionOff();

  CMD_SERIAL.println("▶️ 명령 대기중 (S,C,H,D)");
}

void loop() {
  if (CMD_SERIAL.available()) {
    char cmd = CMD_SERIAL.read();

    CMD_SERIAL.print("수신: "); CMD_SERIAL.println(cmd);

    switch (cmd) {
      case 'S': case 's':
        pickupAndPlace(pose_spade, "스페이드"); break;
      case 'C': case 'c':
        pickupAndPlace(pose_club,  "클로버");   break;
      case 'H': case 'h':
        pickupAndPlace(pose_heart, "하트");     break;
      case 'D': case 'd':
        pickupAndPlace(pose_dia,   "다이아");   break;
      case 'P':  // PASS: 아무 동작 없이 무시
        CMD_SERIAL.println("패스 카드입니다. 무시하고 넘어갑니다.");
        break;
      default:
        CMD_SERIAL.print("알 수 없는 명령: ");
        CMD_SERIAL.println(cmd);
        break;
    }
  }
}
