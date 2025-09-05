#include <Dynamixel2Arduino.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#define DXL_SERIAL   Serial1
#define CMD_SERIAL   Serial2
#define LCD_ADDR     0x27
#define LCD_COLS     20
#define LCD_ROWS     2
const int DXL_DIR_PIN = -1;
const uint8_t motor_ids[] = {1, 2, 3};
const uint8_t num_motors = 3;
#define RELAY_PIN 8
// 포지션 정의
const uint16_t pose_default[] = {167, 181, 170};         // 기본 위치
const uint16_t pose_pickup[]  = {239, 260, 250};         // 픽업(흡착) 위치
const uint16_t pose_spade[]   = {81, 308, 261};         // 스페이드1
const uint16_t pose_spade2[]  = {190, 372, 272};         // 스페이드2
const uint16_t pose_club[]    = {81, 308, 261};         // 클럽1
const uint16_t pose_club2[]   = {168, 315, 340};         // 클럽2
const uint16_t pose_heart[]   = {267, 184, 172};         // 하트1
const uint16_t pose_heart2[]  = {344, 268, 195};         // 하트2
const uint16_t pose_dia[]     = {267, 184, 172};         // 다이아1
const uint16_t pose_dia2[]    = {327, 190, 275};         // 다이아2
// 상태 변수
int count_spade = 0, count_club = 0, count_heart = 0, count_dia = 0;
String pass_card = "";
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
LiquidCrystal_I2C lcd(LCD_ADDR, LCD_COLS, LCD_ROWS);
// 모든 모터 토크 해제
void torqueOffAllMotors() {
  for (int i = 0; i < num_motors; i++)
    dxl.torqueOff(motor_ids[i]);
  CMD_SERIAL.println("모든 모터 토크 해제 완료");
}
// 현재 모터 위치 출력
void printCurrentPositions() {
  CMD_SERIAL.print("모터 위치: ");
  for (int i = 0; i < num_motors; i++) {
    uint16_t pos = dxl.getPresentPosition(motor_ids[i]);
    CMD_SERIAL.print(pos);
    if (i != num_motors - 1) CMD_SERIAL.print(", ");
  }
  CMD_SERIAL.println();
}
// 이동 함수
void smoothMoveToPose(const uint16_t *target, const char* name, uint8_t steps = 20, uint16_t delayMs = 5) {
  CMD_SERIAL.print("이동: "); CMD_SERIAL.println(name);
  uint16_t curr[num_motors];
  for (int i = 0; i < num_motors; i++) curr[i] = dxl.getPresentPosition(motor_ids[i]);
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
void suctionOn()  { digitalWrite(RELAY_PIN, HIGH); }
void suctionOff() { digitalWrite(RELAY_PIN, LOW);  }
void pickupAndPlace(const uint16_t *dest1, const char* label, char symbol) {
  // 1. 기본 위치로 이동
  smoothMoveToPose(pose_default, "기본");   delay(200);
  // 2. 픽업 위치로 이동
  smoothMoveToPose(pose_pickup,  "픽업");   delay(200);
  // 3. 흡착 ON
  suctionOn(); delay(700);
  // 4. 문양1 위치로 이동
  smoothMoveToPose(dest1, label); delay(200);
  // 5. 문양2 위치로 이동 (문양에 따라 분기)
  if (dest1 == pose_spade) {
    smoothMoveToPose(pose_spade2, "스페이드2"); delay(200);
    suctionOff(); delay(1700);
    smoothMoveToPose(pose_spade, "스페이드1(복귀)"); delay(200);
  } else if (dest1 == pose_club) {
    smoothMoveToPose(pose_club2, "클로버2"); delay(200);
    suctionOff(); delay(1700);
    smoothMoveToPose(pose_club, "클로버1(복귀)"); delay(200);
  } else if (dest1 == pose_heart) {
    smoothMoveToPose(pose_heart2, "하트2"); delay(200);
    suctionOff(); delay(1700);
    smoothMoveToPose(pose_heart, "하트1(복귀)"); delay(200);
  } else if (dest1 == pose_dia) {
    smoothMoveToPose(pose_dia2, "다이아2"); delay(200);
    suctionOff(); delay(1700);
    smoothMoveToPose(pose_dia, "다이아1(복귀)"); delay(200);
  }
  // 6. 마지막에 기본 위치로 복귀
  smoothMoveToPose(pose_default, "기본복귀");
  // 분류 개수 증가
  switch (symbol) {
    case 'S': count_spade++; break;
    case 'C': count_club++;  break;
    case 'H': count_heart++; break;
    case 'D': count_dia++;   break;
  }
  updateLCD();
}
void updateLCD() {
  lcd.clear();
  lcd.setCursor(0, 0);
  String displayText = "PASS CARD : " + pass_card;
  if (displayText.length() > 20)
    displayText = displayText.substring(0, 20);  // 20자 초과 방지
  lcd.print(displayText);
  // 두 번째 줄: 한 자리씩(띄어쓰기 포함)
  lcd.setCursor(0, 1);
  lcd.print("S:"); lcd.print(count_spade);
  lcd.print(" C:"); lcd.print(count_club);
  lcd.print(" D:"); lcd.print(count_dia);
  lcd.print(" H:"); lcd.print(count_heart);
}
void setup() {
  CMD_SERIAL.begin(115200);
  dxl.begin(115200);
  dxl.setPortProtocolVersion(1.0);
  for (int i = 0; i < num_motors; i++) {
    dxl.torqueOff(motor_ids[i]);
    dxl.setOperatingMode(motor_ids[i], OP_POSITION);
    dxl.torqueOn(motor_ids[i]);
  }
  pinMode(RELAY_PIN, OUTPUT);
  suctionOff();
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("PASS CARD :");
  lcd.setCursor(0, 1);
  lcd.print("S:0 C:0 D:0 H:0");
  CMD_SERIAL.println("명령 대기중 (S,C,H,D,P)");
}
void loop() {
  if (CMD_SERIAL.available()) {
    String cmd = CMD_SERIAL.readStringUntil('\n');
    cmd.trim();  // ← 줄바꿈, 공백 제거 필수!
    CMD_SERIAL.print("수신: "); CMD_SERIAL.println(cmd);
    if (cmd.length() == 1) {
      char symbol = cmd.charAt(0);
      switch (symbol) {
        case 'S': case 's': pickupAndPlace(pose_spade, "스페이드", 'S'); break;
        case 'C': case 'c': pickupAndPlace(pose_club,  "클로버",   'C'); break;
        case 'H': case 'h': pickupAndPlace(pose_heart, "하트",     'H'); break;
        case 'D': case 'd': pickupAndPlace(pose_dia,   "다이아",   'D'); break;
        case 'R': case 'r':   // <-- 추가
          lcd.clear();
          lcd.setCursor(0, 0);
          lcd.print("LCD RESET");
          lcd.setCursor(0, 1);
          lcd.print("S:0 C:0 D:0 H:0");
          // 필요하면 변수도 리셋!
          count_spade = 0;
          count_club = 0;
          count_heart = 0;
          count_dia = 0;
          pass_card = "";
          CMD_SERIAL.println("LCD 초기화 완료");
          break;
        case 'k': case 'K': torqueOffAllMotors(); break;    // 토크 해제
        case 't': case 'T': printCurrentPositions(); break;  // 좌표 출력
        case 'o': case 'O': suctionOn(); CMD_SERIAL.println("흡착 ON"); break;
        case 'f': case 'F': suctionOff(); CMD_SERIAL.println("흡착 OFF"); break;
        case 'q': case 'Q': // 기본 위치로 이동
          smoothMoveToPose(pose_default, "기본위치");
          CMD_SERIAL.println("기본 위치로 이동 완료");
          break;
        case 'w': case 'W': // 픽업 위치로 이동
          smoothMoveToPose(pose_pickup, "픽업위치");
          CMD_SERIAL.println("픽업 위치로 이동 완료");
          break;
        default:
          CMD_SERIAL.println("알 수 없는 명령"); break;
      }
    } else {
      // 여러 글자인 경우: PASS 카드 표시용 (예: "AS", "10C")
      pass_card = cmd;
      // 디버깅 로그
      CMD_SERIAL.print("pass_card 변수 값: ");
      CMD_SERIAL.println(pass_card);
      updateLCD();
    }
  }
}