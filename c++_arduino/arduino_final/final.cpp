#include <Dynamixel2Arduino.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
// ---------------------------
// [TUNE] 전역 튜닝 파라미터
// ---------------------------
#define LCD_I2C_HZ       100000  // I2C 속도(깨짐 있으면 50000)
#define EASE_STEPS       20
#define EASE_DELAY_MS    5
#define COOLDOWN_MS      1200

// 하드웨어
#define DXL_SERIAL   Serial1
#define CMD_SERIAL   Serial2        // 외부 TTL(라즈베리/PC 등)
#define LCD_ADDR     0x27
#define LCD_COLS     20
#define LCD_ROWS     2
const int DXL_DIR_PIN = -1;
const uint8_t motor_ids[] = {1, 2, 3};
const uint8_t num_motors = 3;
#define RELAY_PIN 8
// 포지션(원본)
const uint16_t pose_default[] = {125, 136, 196};
const uint16_t pose_pickup[]  = {230, 240, 290};
const uint16_t pose_spade[]   = {70, 300, 249};
const uint16_t pose_spade2[]  = {168, 369, 240};
const uint16_t pose_club[]    = {70, 300, 249};
const uint16_t pose_club2[]   = {154, 314, 342};
const uint16_t pose_heart[]   = {299, 181, 172};
const uint16_t pose_heart2[]  = {330, 190, 272};
const uint16_t pose_dia[]     = {299, 181, 172};
const uint16_t pose_dia2[]    = {351, 285, 188};
// 상태
int count_spade = 0, count_club = 0, count_heart = 0, count_dia = 0;
String pass_card = "";
// Busy/쿨다운
volatile bool BUSY = false;
unsigned long lastCmdMs = 0;
char lastSymbol = 0;
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
LiquidCrystal_I2C lcd(LCD_ADDR, LCD_COLS, LCD_ROWS);

const unsigned long HOLD_DUP_MS = 3500;
char lastDoneSymbol = 0;
unsigned long lastDoneMs = 0;
// ----------------- LCD 유틸 -----------------
void lcdSoftClear() {
  lcd.setCursor(0, 0);
  for (int i = 0; i < LCD_COLS; i++) lcd.print(' ');
  lcd.setCursor(0, 1);
  for (int i = 0; i < LCD_COLS; i++) lcd.print(' ');
  lcd.setCursor(0, 0);
  delay(2);
}
void lcdPrintTitleAndCounts() {
  lcd.setCursor(0, 0);
  lcd.print(F("   TRUMP CARD"));
  char line[21];
  snprintf(line, sizeof(line), "S:%d C:%d H:%d D:%d",
           count_spade, count_club, count_heart, count_dia);
  lcd.setCursor(0, 1);
  lcd.print(line);
  int len = strlen(line);
  for (int i = len; i < LCD_COLS; i++) lcd.print(' ');
}
// ----------------- 모터/흡착 유틸 -----------------
void torqueOffAllMotors() {
  for (int i = 0; i < num_motors; i++) dxl.torqueOff(motor_ids[i]);
  CMD_SERIAL.println("모든 모터 토크 해제 완료");
  Serial.println("모든 모터 토크 해제 완료");
}
void printCurrentPositions() {
  CMD_SERIAL.print("모터 위치: ");
  Serial.print("모터 위치: ");
  for (int i = 0; i < num_motors; i++) {
    uint16_t pos = dxl.getPresentPosition(motor_ids[i]);
    CMD_SERIAL.print(pos); Serial.print(pos);
    if (i != num_motors - 1) { CMD_SERIAL.print(", "); Serial.print(", "); }
  }
  CMD_SERIAL.println(); Serial.println();
}
void suctionOn()  { digitalWrite(RELAY_PIN, HIGH); }
void suctionOff() { digitalWrite(RELAY_PIN, LOW);  }
void smoothMoveToPose(const uint16_t *target, const char* name,
                      uint8_t steps = EASE_STEPS, uint16_t delayMs = EASE_DELAY_MS) {
  CMD_SERIAL.print("이동: "); CMD_SERIAL.println(name);
  Serial.print("이동: "); Serial.println(name);
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
// ----------------- 공용 입력 함수(USB/TTL) -----------------
int readCharFromAny() {
  if (CMD_SERIAL.available()) return CMD_SERIAL.read(); // Serial2
  if (Serial.available())     return Serial.read();     // USB Serial
  return -1;
}
// ----------------- 픽업/분류 -----------------
void pickupAndPlace(const uint16_t *dest1, const char* label, char symbol) {
  BUSY = true;   // ★ 시작 래치

  // ====== 아래 순서/딜레이를 고정 값으로 사용 ======
  smoothMoveToPose(pose_default, "기본");   delay(10);
  suctionOn();                               delay(100);
  smoothMoveToPose(pose_pickup,  "픽업");    delay(100);
  smoothMoveToPose(pose_default, "기본");    delay(200);

  // ※ 요청대로: 목적지로 가기 전에 흡착 해제
  suctionOff();                              delay(100);
  smoothMoveToPose(dest1, label);            delay(10);

  if (dest1 == pose_spade) {
    smoothMoveToPose(pose_spade2, "스페이드2");         delay(1000);
    //suctionOff();                                      delay(1700);
    smoothMoveToPose(pose_spade,  "스페이드1(복귀)");   delay(100);
  } else if (dest1 == pose_club) {
    smoothMoveToPose(pose_club2, "클로버2");           delay(1000);
    //suctionOff();                                      delay(1700);
    smoothMoveToPose(pose_club,  "클로버1(복귀)");     delay(100);
  } else if (dest1 == pose_heart) {
    smoothMoveToPose(pose_heart2, "하트2");             delay(1000);
    //suctionOff();                                      delay(1700);
    smoothMoveToPose(pose_heart, "하트1(복귀)");        delay(100);
  } else if (dest1 == pose_dia) {
    smoothMoveToPose(pose_dia2, "다이아2");             delay(1000);
    //suctionOff();                                      delay(1700);
    smoothMoveToPose(pose_dia,  "다이아1(복귀)");       delay(100);
  }

  smoothMoveToPose(pose_default, "기본복귀");

  switch (symbol) {
    case 'S': count_spade++; break;
    case 'C': count_club++;  break;
    case 'H': count_heart++; break;
    case 'D': count_dia++;   break;
  }
  lcdSoftClear();
  lcdPrintTitleAndCounts();
  BUSY = false;

  lastDoneSymbol = symbol;
  lastDoneMs = millis();
}
// ----------------- 셋업 -----------------
void setup() {
  // USB 시리얼(Arduino Serial Monitor)
  Serial.begin(115200);
  // I2C 초기화 + 속도
  Wire.begin();
  Wire.setClock(LCD_I2C_HZ);
  // Dynamixel/명령 포트
  CMD_SERIAL.begin(115200);     // Serial2
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
  delay(5);
  lcdSoftClear();
  lcdPrintTitleAndCounts();
  Serial.println("명령 대기중 (S/C/H/D, K/T/O/F/Q/W, R)");
  CMD_SERIAL.println("명령 대기중 (S/C/H/D, K/T/O/F/Q/W, R)");
}
// ----------------- 루프 -----------------
void loop() {
  // ★ BUSY 중에도 'R'은 즉시 처리(카운트/LCD 초기화만 수행)
  if (BUSY) {
    int c = readCharFromAny();
    if (c != -1) {
      char ch = (char)c;
      if (ch >= 'a' && ch <= 'z') ch -= 32; // to upper
      if (ch == 'R') {
        count_spade = count_club = count_heart = count_dia = 0;
        pass_card = "";
        lcdSoftClear();
        lcdPrintTitleAndCounts();
        Serial.println("LCD 초기화(R)");
        CMD_SERIAL.println("LCD 초기화(R)");
      }
    }
    return;
  }
  int c = readCharFromAny();
  if (c == -1) return;
  char ch = (char)c;
  if (ch == '\r' || ch == '\n' || ch == ' ') return;
  if (ch >= 'a' && ch <= 'z') ch -= 32;
  unsigned long now = millis();
  if (now - lastCmdMs < COOLDOWN_MS) { Serial.println("IGN: cooldown"); CMD_SERIAL.println("IGN: cooldown"); return; }
  if ((ch == lastSymbol) && (now - lastCmdMs < COOLDOWN_MS + 200)) { Serial.println("IGN: duplicate"); CMD_SERIAL.println("IGN: duplicate"); return; }
  Serial.print("수신: "); Serial.println(ch);
  CMD_SERIAL.print("수신: "); CMD_SERIAL.println(ch);
  if (ch == lastDoneSymbol && (millis() - lastDoneMs) < HOLD_DUP_MS) {
    Serial.println("IGN: repeat-hold");
    CMD_SERIAL.println("IGN: repeat-hold");
    return;
  }
  switch (ch) {
    case 'S': pickupAndPlace(pose_spade, "스페이드", 'S'); break;
    case 'C': pickupAndPlace(pose_club,  "클로버",   'C'); break;
    case 'H': pickupAndPlace(pose_heart, "하트",     'H'); break;
    case 'D': pickupAndPlace(pose_dia,   "다이아",   'D'); break;
    case 'K': torqueOffAllMotors(); break;
    case 'T': printCurrentPositions(); break;
    case 'O': suctionOn();  Serial.println("흡착 ON");  CMD_SERIAL.println("흡착 ON");  break;
    case 'F': suctionOff(); Serial.println("흡착 OFF"); CMD_SERIAL.println("흡착 OFF"); break;
    case 'Q': smoothMoveToPose(pose_default, "기본위치"); Serial.println("기본 위치로 이동 완료"); CMD_SERIAL.println("기본 위치로 이동 완료"); break;
    case 'W': smoothMoveToPose(pose_pickup,  "픽업위치");  Serial.println("픽업 위치로 이동 완료");  CMD_SERIAL.println("픽업 위치로 이동 완료");  break;
    case 'R': // 초기 화면으로 리셋
      count_spade = count_club = count_heart = count_dia = 0;
      pass_card = "";
      lcdSoftClear();
      lcdPrintTitleAndCounts();
      Serial.println("LCD 초기화 완료");
      CMD_SERIAL.println("LCD 초기화 완료");
      break;
    default:
      Serial.println("알 수 없는 명령");
      CMD_SERIAL.println("알 수 없는 명령");
      break;
  }
  lastSymbol = ch;
  lastCmdMs  = now;
}