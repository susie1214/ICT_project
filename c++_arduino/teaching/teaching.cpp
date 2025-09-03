#include <Dynamixel2Arduino.h>

#define DXL_SERIAL   Serial1
const int DXL_DIR_PIN = -1;
const uint8_t motor_ids[] = {1, 2, 3};
const uint8_t num_motors = 3;

#define RELAY_PIN 8  // 릴레이(흡착기) 제어 핀 번호

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

void setup() {
  Serial.begin(115200);        // 시리얼 모니터용
  dxl.begin(115200);           // 다이나믹셀 통신 속도
  dxl.setPortProtocolVersion(1.0);

  // 모터 준비 (토크 ON, 위치모드)
  for (int i = 0; i < num_motors; i++) {
    dxl.torqueOff(motor_ids[i]);
    dxl.setOperatingMode(motor_ids[i], OP_POSITION);
    dxl.torqueOn(motor_ids[i]);
  }

  pinMode(RELAY_PIN, OUTPUT);   // 릴레이 핀 출력 설정
  digitalWrite(RELAY_PIN, LOW); // 초기 흡착 OFF

  Serial.println("s: 현재 좌표 출력, k: 토크 해제, o: 흡착 ON, f: 흡착 OFF");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();

    if (cmd == 's' || cmd == 'S') {
      // 현재 좌표 출력
      Serial.print("현재 좌표: ");
      for (int i = 0; i < num_motors; i++) {
        uint16_t pos = dxl.getPresentPosition(motor_ids[i]);
        Serial.print("M"); Serial.print(motor_ids[i]);
        Serial.print(":"); Serial.print(pos);
        if (i < num_motors - 1) Serial.print(", ");
      }
      Serial.println();
    }
    else if (cmd == 'k' || cmd == 'K') {
      // 토크 해제
      for (int i = 0; i < num_motors; i++) {
        dxl.torqueOff(motor_ids[i]);
      }
      Serial.println("토크 해제 완료!");
    }
    else if (cmd == 'o' || cmd == 'O') {
      // 흡착 ON
      digitalWrite(RELAY_PIN, HIGH);
      Serial.println("흡착 ON!");
    }
    else if (cmd == 'f' || cmd == 'F') {
      // 흡착 OFF
      digitalWrite(RELAY_PIN, LOW);
      Serial.println("흡착 OFF!");
    }
  }
}
