// ------------------------------------------------------------
// OpenRB-150 (Arduino) 스케치: 시리얼 1바이트 명령으로
// 델타로봇(AX-12 x3)과 흡착 릴레이를 제어해 카드를 분류/이동
// - ‘S/C/H/D’ 수신 → 티칭 좌표로 부드럽게 이동(smoothMoveToPose)
// - BUSY/쿨다운/중복 필터로 단발 동작 보장
// - LCD에 통계 표시
// ------------------------------------------------------------

#include <Dynamixel2Arduino.h>  // AX-12 제어 라이브러리
#include <Wire.h>               // I2C (LCD)
#include <LiquidCrystal_I2C.h>  // I2C LCD

// ===== 시리얼 포트/하드웨어 매핑 =====
#define DXL_SERIAL   Serial1     // 다이나믹셀(AX-12) 연결 포트
#define CMD_SERIAL   Serial2     // PC(맥북)와 통신할 명령 포트(UART)
#define LCD_ADDR     0x27        // I2C LCD 주소
#define LCD_COLS     20          // LCD 가로 글자수
#define LCD_ROWS     2           // LCD 세로 줄수

const int DXL_DIR_PIN = -1;      // Half-duplex 방향제어 핀(AX-12는 -1 사용)
const uint8_t motor_ids[] = {1, 2, 3}; // AX-12 세 모터의 ID
const uint8_t num_motors = 3;

#define RELAY_PIN 8              // 흡착기(진공펌프/솔레노이드) 릴레이 제어 핀

// ===== 티칭(목표) 포즈 정의 =====
// 각 배열은 모터 1,2,3의 목표 위치값(틱)이며, 프로젝트에서 미리 티칭해 둔 값
const uint16_t pose_default[] = {125, 136, 196};         // 기본 위치(대기)
const uint16_t pose_pickup[]  = {230, 240, 290};         // 픽업(흡착) 위치
const uint16_t pose_spade[]   = {70, 300, 249};          // 스페이드 투입 위치(1)
const uint16_t pose_spade2[]  = {168, 369, 240};         // 스페이드 경유(2)
const uint16_t pose_club[]    = {70, 300, 249};          // 클로버(1)
const uint16_t pose_club2[]   = {154, 314, 342};         // 클로버(2)
const uint16_t pose_heart[]   = {299, 181, 172};         // 하트(1)
const uint16_t pose_heart2[]  = {330, 190, 272};         // 하트(2)
const uint16_t pose_dia[]     = {299, 181, 172};         // 다이아(1)
const uint16_t pose_dia2[]    = {351, 285, 188};         // 다이아(2)

// ===== 카운트/표시용 상태 =====
int count_spade = 0, count_club = 0, count_heart = 0, count_dia = 0;
String pass_card = "";  // 최근 처리 카드 텍스트(원한다면 갱신해 사용)

// ===== 입력 안정화: BUSY/쿨다운/중복 필터 =====
volatile bool BUSY = false;                  // 동작 중이면 추가 명령 무시
unsigned long lastCmdMs = 0;                 // 마지막 명령 처리 시각(ms)
const unsigned long COOLDOWN_MS = 1200;      // 이 시간 내 재실행 금지
char lastSymbol = 0;                          // 직전 명령 문자 저장

// ===== 디바이스 인스턴스 =====
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
LiquidCrystal_I2C lcd(LCD_ADDR, LCD_COLS, LCD_ROWS);

// ------------------------------------------------------------
// 유틸 함수들
// ------------------------------------------------------------

// 모든 모터 토크 OFF (수동 이동/안전 점검용)
void torqueOffAllMotors() {
  for (int i = 0; i < num_motors; i++) dxl.torqueOff(motor_ids[i]);
  CMD_SERIAL.println("모든 모터 토크 해제 완료");
}

// 현재 모터 위치를 시리얼로 출력(티칭/디버깅에 유용)
void printCurrentPositions() {
  CMD_SERIAL.print("모터 위치: ");
  for (int i = 0; i < num_motors; i++) {
    uint16_t pos = dxl.getPresentPosition(motor_ids[i]);
    CMD_SERIAL.print(pos);
    if (i != num_motors - 1) CMD_SERIAL.print(", ");
  }
  CMD_SERIAL.println();
}

// 흡착기 ON/OFF (릴레이)
void suctionOn()  { digitalWrite(RELAY_PIN, HIGH); }
void suctionOff() { digitalWrite(RELAY_PIN, LOW);  }

// 부드럽게 목표 포즈로 이동(코사인 이징)
// - target: 목표 각도 배열(티칭 값)
// - name  : 로그용 이름
// - steps : 보간 스텝 수(클수록 더 부드러움)
// - delayMs: 스텝 간 지연(속도 제어)
void smoothMoveToPose(const uint16_t *target, const char* name, uint8_t steps = 20, uint16_t delayMs = 5) {
  CMD_SERIAL.print("이동: "); CMD_SERIAL.println(name);

  // 현재 각도 읽어오기
  uint16_t curr[num_motors];
  for (int i = 0; i < num_motors; i++) curr[i] = dxl.getPresentPosition(motor_ids[i]);

  // 코사인 이징으로 0→1 곡선 생성하여 보간 이동
  for (int s = 1; s <= steps; s++) {
    float t = (float)s / steps;
    float ease = (1 - cos(t * PI)) / 2;  // 0→1, 가감속 느낌
    for (int i = 0; i < num_motors; i++) {
      uint16_t pos = curr[i] + (target[i] - curr[i]) * ease;
      dxl.setGoalPosition(motor_ids[i], pos);
    }
    delay(delayMs);
  }
}

// ------------------------------------------------------------
// 픽업/분류 동작 시퀀스
// - dest1: 해당 심볼의 1번 투입 위치(필요 시 dest2로 왕복)
// - label: 로그용 라벨
// - symbol: 'S','C','H','D' (카운트/UI에 사용)
// ------------------------------------------------------------
void pickupAndPlace(const uint16_t *dest1, const char* label, char symbol) {
  BUSY = true;   // ★ 시작: 동작 중 래치

  // ===== 원래 쓰던 순서/딜레이 유지(현장 보정값 포함) =====
  smoothMoveToPose(pose_default, "기본");   delay(3300);  // 카드 유입 대기
  suctionOn(); delay(100);                                  // 흡착 시작
  smoothMoveToPose(pose_pickup,  "픽업");   delay(100);     // 픽업 포즈
  smoothMoveToPose(pose_default, "기본");   delay(200);     // 들고 복귀
  suctionOff(); delay(100);                                   // (옵션) 이 타이밍 해제
  smoothMoveToPose(dest1, label);             delay(200);     // 목적지 1로 이동

  // 필요 시 목적지 2 경유(프로파일에 맞춘 왕복)
  if (dest1 == pose_spade) {
    smoothMoveToPose(pose_spade2, "스페이드2"); delay(500);
    // suctionOff(); delay(1700);  // 필요하면 이 지점에 해제
    smoothMoveToPose(pose_spade, "스페이드1(복귀)"); delay(500);
  } else if (dest1 == pose_club) {
    smoothMoveToPose(pose_club2, "클로버2"); delay(500);
    // suctionOff(); delay(1700);
    smoothMoveToPose(pose_club, "클로버1(복귀)"); delay(500);
  } else if (dest1 == pose_heart) {
    smoothMoveToPose(pose_heart2, "하트2"); delay(500);
    // suctionOff(); delay(1700);
    smoothMoveToPose(pose_heart, "하트1(복귀)"); delay(200);
  } else if (dest1 == pose_dia) {
    smoothMoveToPose(pose_dia2, "다이아2"); delay(200);
    // suctionOff(); delay(1700);
    smoothMoveToPose(pose_dia, "다이아1(복귀)"); delay(200);
  }

  // 기본 위치로 복귀(다음 동작 대기)
  smoothMoveToPose(pose_default, "기본복귀");

  // ===== 통계/표시 갱신 =====
  switch (symbol) {
    case 'S': count_spade++; break;
    case 'C': count_club++;  break;
    case 'H': count_heart++; break;
    case 'D': count_dia++;   break;
  }

  // LCD 1줄: PASS CARD 텍스트, 2줄: S/C/D/H 카운트
  lcd.clear();
  lcd.setCursor(0, 0);
  {
    String displayText = "PASS CARD : " + pass_card;
    if (displayText.length() > 20) displayText = displayText.substring(0, 20);
    lcd.print(displayText);
  }
  lcd.setCursor(0, 1);
  lcd.print("S:"); lcd.print(count_spade);
  lcd.print(" C:"); lcd.print(count_club);
  lcd.print(" D:"); lcd.print(count_dia);
  lcd.print(" H:"); lcd.print(count_heart);

  BUSY = false;  // ★ 종료: READY 상태로 복귀
}

// ------------------------------------------------------------
// 초기화(Setup)
// ------------------------------------------------------------
void setup() {
  // PC(맥북)와의 명령 시리얼
  CMD_SERIAL.begin(115200);

  // 다이나믹셀 포트 초기화 + 프로토콜 버전 설정
  dxl.begin(115200);
  dxl.setPortProtocolVersion(1.0);

  // 각 모터를 위치제어 모드로 설정 후 토크 ON
  for (int i = 0; i < num_motors; i++) {
    dxl.torqueOff(motor_ids[i]);
    dxl.setOperatingMode(motor_ids[i], OP_POSITION);
    dxl.torqueOn(motor_ids[i]);
  }

  // 흡착 릴레이 핀 초기화
  pinMode(RELAY_PIN, OUTPUT);
  suctionOff();  // 안전상 기본 OFF

  // LCD 초기화 및 안내 표시
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0); lcd.print("PASS CARD :");
  lcd.setCursor(0, 1); lcd.print("S:0 C:0 D:0 H:0");

  // 콘솔 안내
  CMD_SERIAL.println("명령 대기중 (S/C/H/D, K/T/O/F/Q/W, R)");
}

// ------------------------------------------------------------
// 메인 루프: 1바이트 명령 처리
// ------------------------------------------------------------
void loop() {
  // 동작 중엔 새 입력(버퍼) 즉시 비우고 무시 → 큐 적체 방지
  if (BUSY) {
    while (CMD_SERIAL.available()) CMD_SERIAL.read();
    return;
  }

  // 명령 수신이 있으면 1바이트 읽어 처리
  if (CMD_SERIAL.available()) {
    char ch = CMD_SERIAL.read();

    // 개행/공백 무시
    if (ch == '\r' || ch == '\n' || ch == ' ') return;

    // 소문자 → 대문자 통일
    if (ch >= 'a' && ch <= 'z') ch -= 32;

    unsigned long now = millis();

    // 쿨다운: 마지막 명령 이후 COOLDOWN_MS 이내면 무시
    if (now - lastCmdMs < COOLDOWN_MS) {
      CMD_SERIAL.println("IGN: cooldown");
      return;
    }

    // 같은 심볼을 너무 빨리 반복하면 무시(채터링 억제)
    if ((ch == lastSymbol) && (now - lastCmdMs < COOLDOWN_MS + 200)) {
      CMD_SERIAL.println("IGN: duplicate");
      return;
    }

    // 수신 로그
    CMD_SERIAL.print("수신: "); CMD_SERIAL.println(ch);

    // 명령 분기
    switch (ch) {
      case 'S': pickupAndPlace(pose_spade, "스페이드", 'S'); break;
      case 'C': pickupAndPlace(pose_club,  "클로버",   'C'); break;
      case 'H': pickupAndPlace(pose_heart, "하트",     'H'); break;
      case 'D': pickupAndPlace(pose_dia,   "다이아",   'D'); break;

      case 'K': torqueOffAllMotors(); break;     // 모든 모터 토크 해제
      case 'T': printCurrentPositions(); break;  // 현재 좌표 출력
      case 'O': suctionOn();  CMD_SERIAL.println("흡착 ON");  break;
      case 'F': suctionOff(); CMD_SERIAL.println("흡착 OFF"); break;
      case 'Q': smoothMoveToPose(pose_default, "기본위치"); CMD_SERIAL.println("기본 위치로 이동 완료"); break;
      case 'W': smoothMoveToPose(pose_pickup,  "픽업위치");  CMD_SERIAL.println("픽업 위치로 이동 완료");  break;

      case 'R': // LCD/카운트 리셋
        lcd.clear();
        lcd.setCursor(0, 0); lcd.print("LCD RESET");
        lcd.setCursor(0, 1); lcd.print("S:0 C:0 D:0 H:0");
        count_spade = count_club = count_heart = count_dia = 0;
        pass_card = "";
        CMD_SERIAL.println("LCD 초기화 완료");
        break;

      default:
        CMD_SERIAL.println("알 수 없는 명령");
        break;
    }

    // 중복 억제/쿨다운 관리를 위한 타임스탬프 갱신
    lastSymbol = ch;
    lastCmdMs  = now;
  }
}
