#include <Dynamixel2Arduino.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <string.h>   // strtok
#include <math.h>

#define DXL_SERIAL   Serial1
#define CMD_SERIAL   Serial2
#define LCD_ADDR     0x27
#define LCD_COLS     20
#define LCD_ROWS     2

const int DXL_DIR_PIN = -1;
const uint8_t motor_ids[] = {1, 2, 3};
const uint8_t num_motors = 3;

#define RELAY_PIN 8

// ===== 포지션 정의 (원본 그대로) =====
const uint16_t pose_default[] = {125, 136, 196};         // 기본 위치
const uint16_t pose_pickup[]  = {230, 240, 290};         // 픽업(흡착) 위치
const uint16_t pose_spade[]   = {70, 300, 249};          // 스페이드1
const uint16_t pose_spade2[]  = {168, 369, 240};         // 스페이드2
const uint16_t pose_club[]    = {70, 300, 249};          // 클럽1
const uint16_t pose_club2[]   = {154, 314, 342};         // 클럽2
const uint16_t pose_heart[]   = {299, 181, 172};         // 하트1
const uint16_t pose_heart2[]  = {330, 190, 272};         // 하트2
const uint16_t pose_dia[]     = {299, 181, 172};         // 다이아1
const uint16_t pose_dia2[]    = {351, 285, 188};         // 다이아2

// ===== 모델/카운트 구성 =====
enum ModelType { MODEL_CARD=0, MODEL_RECYCLE=1, MODEL_BEVERAGE=2, MODEL_BEEF=3 };
const char* MODEL_NAME[] = { "CARD", "RECYCLE", "BEVERAGE", "BEEF" };

struct Counts4 { int a=0, b=0, c=0, d=0; };
Counts4 cnt_card, cnt_recycle, cnt_beverage, cnt_beef;

ModelType ACTIVE_MODEL = MODEL_CARD;  // 기본 표시 모델

// ===== Busy / Home / 쿨다운 / 중복 =====
volatile bool BUSY = false;           // 동작 중 입력 차단(버퍼 즉시 폐기)
volatile bool AT_HOME = true;         // 디폴트 위치 래치
unsigned long lastCmdMs = 0;
const unsigned long COOLDOWN_MS = 1200;
char lastSymbol = 0;

const uint16_t HOME_TOL = 6;          // 디폴트 위치 판정 허용오차 (틱)
const uint16_t POS_MIN = 0;           // 안전 클램프 (필요시 수정)
const uint16_t POS_MAX = 1023;

inline uint16_t _clampPos(int v) {
  if (v < (int)POS_MIN) return POS_MIN;
  if (v > (int)POS_MAX) return POS_MAX;
  return (uint16_t)v;
}


// ===== 하드웨어 =====
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
LiquidCrystal_I2C lcd(LCD_ADDR, LCD_COLS, LCD_ROWS);

// ----------------- 유틸 -----------------
uint16_t clampU16(int v){ if(v<POS_MIN) v=POS_MIN; if(v>POS_MAX) v=POS_MAX; return (uint16_t)v; }

void torqueOffAllMotors() {
  for (int i = 0; i < num_motors; i++) dxl.torqueOff(motor_ids[i]);
  CMD_SERIAL.println("모든 모터 토크 해제 완료");
}

void printCurrentPositions() {
  CMD_SERIAL.print("모터 위치: ");
  for (int i = 0; i < num_motors; i++) {
    uint16_t pos = dxl.getPresentPosition(motor_ids[i]);
    CMD_SERIAL.print(pos);
    if (i != num_motors - 1) CMD_SERIAL.print(", ");
  }
  CMD_SERIAL.println();
}

bool near(uint16_t a, uint16_t b, uint16_t tol){ return (a > b ? (a-b) : (b-a)) <= tol; }

bool isAtPose(const uint16_t *pose, uint16_t tol=HOME_TOL){
  for(int i=0;i<num_motors;i++){
    uint16_t pos = dxl.getPresentPosition(motor_ids[i]);
    if(!near(pos, pose[i], tol)) return false;
  }
  return true;
}

void waitUntilHome(uint16_t timeout_ms=2500){
  unsigned long t0 = millis();
  while(millis()-t0 < timeout_ms){
    if(isAtPose(pose_default)) return;
    delay(20);
  }
}

void suctionOn()  { digitalWrite(RELAY_PIN, HIGH); }
void suctionOff() { digitalWrite(RELAY_PIN, LOW);  }

void smoothMoveToPose(const uint16_t *target, const char* name, uint8_t steps = 60, uint16_t delayMs = 8) {
  CMD_SERIAL.print("이동: "); CMD_SERIAL.println(name);
  uint16_t curr[num_motors];
  for (int i = 0; i < num_motors; i++) curr[i] = dxl.getPresentPosition(motor_ids[i]);

  // cos 이징으로 점진 이동 (ease-in-out)
  for (int s = 1; s <= steps; s++) {
    float t = (float)s / (float)steps;
    float ease = (1.0f - cosf(t * PI)) * 0.5f; // 0~1
    for (int i = 0; i < num_motors; i++) {
      float interp = (float)curr[i] + ((float)target[i] - (float)curr[i]) * ease;
      uint16_t pos = _clampPos((int)(interp + 0.5f));  // 반올림 + 안전 클램프
      dxl.setGoalPosition(motor_ids[i], pos);
    }
    delay(delayMs);
  }
}

// ===== LCD 표시 =====
void lcdPrintCountsLine() {
  lcd.setCursor(0, 1);
  switch (ACTIVE_MODEL) {
    case MODEL_CARD:
      lcd.print("S:"); lcd.print(cnt_card.a);
      lcd.print(" C:"); lcd.print(cnt_card.b);
      lcd.print(" H:"); lcd.print(cnt_card.c);
      lcd.print(" D:"); lcd.print(cnt_card.d);
      break;
    case MODEL_RECYCLE:  // PL GL MT CB
      lcd.print("PL:"); lcd.print(cnt_recycle.a);
      lcd.print(" GL:"); lcd.print(cnt_recycle.b);
      lcd.print(" MT:"); lcd.print(cnt_recycle.c);
      lcd.print(" CB:"); lcd.print(cnt_recycle.d);
      break;
    case MODEL_BEVERAGE: // CO DW HT LT
      lcd.print("CO:"); lcd.print(cnt_beverage.a);
      lcd.print(" DW:"); lcd.print(cnt_beverage.b);
      lcd.print(" HT:"); lcd.print(cnt_beverage.c);
      lcd.print(" LT:"); lcd.print(cnt_beverage.d);
      break;
    case MODEL_BEEF:     // 1++ 1+ 1 2
      lcd.print("1++:"); lcd.print(cnt_beef.a);
      lcd.print(" 1+:");  lcd.print(cnt_beef.b);
      lcd.print(" 1:");   lcd.print(cnt_beef.c);
      lcd.print(" 2:");   lcd.print(cnt_beef.d);
      break;
  }
}

void updateLCD() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(MODEL_NAME[ACTIVE_MODEL]);  // 1줄: 모델명
  lcdPrintCountsLine();                 // 2줄: 카운트
}

void resetCountsFor(ModelType m) {
  switch (m) {
    case MODEL_CARD:     cnt_card      = Counts4(); break;
    case MODEL_RECYCLE:  cnt_recycle   = Counts4(); break;
    case MODEL_BEVERAGE: cnt_beverage  = Counts4(); break;
    case MODEL_BEEF:     cnt_beef      = Counts4(); break;
  }
}

void setActiveModel(ModelType m, bool alsoReset=true) {
  ACTIVE_MODEL = m;
  if (alsoReset) resetCountsFor(m);
  updateLCD();
  CMD_SERIAL.print("모델 전환: ");
  CMD_SERIAL.println(MODEL_NAME[ACTIVE_MODEL]);

  CMD_SERIAL.println("READY");
}

// ===== 분류 카운트(현재 모델에 맞춰 증가) =====
void addCountBySymbol(char symbol) {
  switch (ACTIVE_MODEL) {
    case MODEL_CARD:
      if (symbol=='S') cnt_card.a++;
      else if (symbol=='C') cnt_card.b++;
      else if (symbol=='H') cnt_card.c++;
      else if (symbol=='D') cnt_card.d++;
      break;
    case MODEL_RECYCLE:   // S->PL, C->GL, H->MT, D->CB
      if (symbol=='S') cnt_recycle.a++;
      else if (symbol=='C') cnt_recycle.b++;
      else if (symbol=='H') cnt_recycle.c++;
      else if (symbol=='D') cnt_recycle.d++;
      break;
    case MODEL_BEVERAGE:  // S->CO, C->DW, H->HT, D->LT
      if (symbol=='S') cnt_beverage.a++;
      else if (symbol=='C') cnt_beverage.b++;
      else if (symbol=='H') cnt_beverage.c++;
      else if (symbol=='D') cnt_beverage.d++;
      break;
    case MODEL_BEEF:      // S->1++, C->1+, H->1, D->2
      if (symbol=='S') cnt_beef.a++;
      else if (symbol=='C') cnt_beef.b++;
      else if (symbol=='H') cnt_beef.c++;
      else if (symbol=='D') cnt_beef.d++;
      break;
  }
}

// ===== 티칭 포즈 전체 출력 =====
void printTeachPoses(){
  auto printPose=[&](const char* name, const uint16_t* p){
    CMD_SERIAL.print(name); CMD_SERIAL.print(": ");
    CMD_SERIAL.print(p[0]); CMD_SERIAL.print(", ");
    CMD_SERIAL.print(p[1]); CMD_SERIAL.print(", ");
    CMD_SERIAL.print(p[2]); CMD_SERIAL.println();
  };
  CMD_SERIAL.println("=== TEACH POSES ===");
  printPose("default", pose_default);
  printPose("pickup ", pose_pickup);
  printPose("spade  ", pose_spade);
  printPose("spade2 ", pose_spade2);
  printPose("club   ", pose_club);
  printPose("club2  ", pose_club2);
  printPose("heart  ", pose_heart);
  printPose("heart2 ", pose_heart2);
  printPose("dia    ", pose_dia);
  printPose("dia2   ", pose_dia2);
  CMD_SERIAL.println("===================");
}

// ===== 수동 좌표 이동 (안전: 이동 후 기본위치 자동 복귀) =====
bool parseTripleToPose(const String& s, uint16_t* out3){
  // 구분자(콤마/세미콜론/콜론/스페이스) 통일
  String t = s;
  for (uint16_t i=0;i<t.length();i++){
    char c = t[i];
    if (c==',' || c==';' || c==':' || c=='\t') t.setCharAt(i,' ');
  }
  char buf[96]; t.toCharArray(buf, sizeof(buf));
  int vals[3]; int n=0;
  char* tok = strtok(buf, " ");
  while(tok && n<3){
    vals[n++] = atoi(tok);
    tok = strtok(NULL, " ");
  }
  if(n!=3) return false;
  out3[0] = clampU16(vals[0]);
  out3[1] = clampU16(vals[1]);
  out3[2] = clampU16(vals[2]);
  return true;
}

void moveManualAndReturnHome(const uint16_t* target){
  BUSY = true; AT_HOME = false;
  CMD_SERIAL.print("MANUAL-> ");
  CMD_SERIAL.print(target[0]); CMD_SERIAL.print(",");
  CMD_SERIAL.print(target[1]); CMD_SERIAL.print(",");
  CMD_SERIAL.print(target[2]); CMD_SERIAL.println();

  smoothMoveToPose(target, "수동이동");
  delay(150);  // 머무름 짧게
  smoothMoveToPose(pose_default, "기본복귀");
  waitUntilHome(2500);
  delay(80);
  updateLCD();
  AT_HOME = true; BUSY = false;
  CMD_SERIAL.println("READY");
}

// ----------------- 픽업/분류 (동작/딜레이 유지 + 래치 강화) -----------------
void pickupAndPlace(const uint16_t *dest1, const char* label, char symbol) {
  BUSY = true;
  AT_HOME = false;
  CMD_SERIAL.println("BUSY");

  // ====== 원본 시퀀스/딜레이 그대로 ======
  smoothMoveToPose(pose_default, "기본");   delay(5500);
  suctionOn(); delay(100);
  smoothMoveToPose(pose_pickup,  "픽업");   delay(100);
  smoothMoveToPose(pose_default, "기본");   delay(200);
  suctionOff(); delay(200);
  smoothMoveToPose(dest1, label); delay(200);

  if (dest1 == pose_spade) {
    smoothMoveToPose(pose_spade2, "스페이드2"); delay(200);
    //suctionOff(); delay(1700);
    smoothMoveToPose(pose_spade, "스페이드1(복귀)"); delay(200);
  } else if (dest1 == pose_club) {
    smoothMoveToPose(pose_club2, "클로버2"); delay(200);
    //suctionOff(); delay(1700);
    smoothMoveToPose(pose_club, "클로버1(복귀)"); delay(200);
  } else if (dest1 == pose_heart) {
    smoothMoveToPose(pose_heart2, "하트2"); delay(200);
    //suctionOff(); delay(1700);
    smoothMoveToPose(pose_heart, "하트1(복귀)"); delay(200);
  } else if (dest1 == pose_dia) {
    smoothMoveToPose(pose_dia2, "다이아2"); delay(200);
    //suctionOff(); delay(1700);
    smoothMoveToPose(pose_dia, "다이아1(복귀)"); delay(200);
  }

  smoothMoveToPose(pose_default, "기본복귀");

  // 디폴트 위치 도착 확인(정착 대기)
  waitUntilHome(2500);
  delay(80); // 소폭 안정화

  // === 카운트 & LCD ===
  addCountBySymbol(symbol);
  updateLCD();

  AT_HOME = true;
  BUSY = false;
  CMD_SERIAL.println("READY");  // 호스트가 이 신호보고 다음 패킷 보내도록 추천
}

// ----------------- 셋업 -----------------
void setup() {
  CMD_SERIAL.begin(115200);
  CMD_SERIAL.setTimeout(300);   // readStringUntil 사용 대비
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
  setActiveModel(MODEL_CARD, true);  // 초기 LCD: "CARD"

  AT_HOME = isAtPose(pose_default);
  BUSY = false;
  CMD_SERIAL.println("READY");
  CMD_SERIAL.println("모델전환: '1'=CARD, '2'=RECYCLE, '3'=BEVERAGE, '4'=BEEF");
  CMD_SERIAL.println("동작: 'S'/'C'/'H'/'D'  | 'P'=티칭포즈 출력 | 'M x,y,z'=수동이동(후 자동복귀)");
  CMD_SERIAL.println("기타: K/T/O/F/Q/W | 'R'=현재 모델 카운트 리셋");
}

// ----------------- 루프 (입력 처리) -----------------
void loop() {
  // 동작 중이거나 아직 홈이 아니면 — 버퍼 즉시 비우고 리턴(큐 방지)
  if (BUSY || !AT_HOME) {
    while (CMD_SERIAL.available()) CMD_SERIAL.read();
    return;
  }

  if (CMD_SERIAL.available()) {
    char ch = CMD_SERIAL.read();

    // 공백/개행/캐리지리턴 무시
    if (ch == '\r' || ch == '\n' || ch == ' ') return;

    // 모델 전환(숫자 1~4)
    if (ch=='1'){ setActiveModel(MODEL_CARD, true); return; }
    if (ch=='2'){ setActiveModel(MODEL_RECYCLE, true); return; }
    if (ch=='3'){ setActiveModel(MODEL_BEVERAGE, true); return; }
    if (ch=='4'){ setActiveModel(MODEL_BEEF, true); return; }

    // 'P' = 티칭 포즈 전체 출력
    if (ch=='P' || ch=='p'){ printTeachPoses(); return; }

    // 'M' = 수동 좌표 이동: M 120,240,310
    if (ch=='M' || ch=='m'){
      String rest = CMD_SERIAL.readStringUntil('\n');
      rest.trim();
      uint16_t manual[3];
      if(parseTripleToPose(rest, manual)){
        moveManualAndReturnHome(manual);  // 이동 후 자동 기본복귀 + READY
      }else{
        CMD_SERIAL.print("입력형식 오류. 예) M 120,240,310  또는  M 120 240 310  | 허용범위 ");
        CMD_SERIAL.print(POS_MIN); CMD_SERIAL.print("~"); CMD_SERIAL.println(POS_MAX);
      }
      return;
    }

    // 영문 소문자 → 대문자
    if (ch >= 'a' && ch <= 'z') ch -= 32;

    unsigned long now = millis();
    // 쿨다운(선택)
    if (now - lastCmdMs < COOLDOWN_MS) {
      CMD_SERIAL.println("IGN: cooldown");
      return;
    }
    // 같은 심볼 빠르게 반복 방지(선택)
    if ((ch == lastSymbol) && (now - lastCmdMs < COOLDOWN_MS + 200)) {
      CMD_SERIAL.println("IGN: duplicate");
      return;
    }

    CMD_SERIAL.print("수신: "); CMD_SERIAL.println(ch);

    switch (ch) {
      case 'S': pickupAndPlace(pose_spade, "스페이드", 'S'); break;
      case 'C': pickupAndPlace(pose_club,  "클로버",   'C'); break;
      case 'H': pickupAndPlace(pose_heart, "하트",     'H'); break;
      case 'D': pickupAndPlace(pose_dia,   "다이아",   'D'); break;

      case 'K': torqueOffAllMotors(); break;
      case 'T': printCurrentPositions(); break;
      case 'O': suctionOn();  CMD_SERIAL.println("흡착 ON");  break;
      case 'F': suctionOff(); CMD_SERIAL.println("흡착 OFF"); break;
      case 'Q': smoothMoveToPose(pose_default, "기본위치"); CMD_SERIAL.println("기본 위치로 이동 완료"); break;
      case 'W': smoothMoveToPose(pose_pickup,  "픽업위치");  CMD_SERIAL.println("픽업 위치로 이동 완료");  break;

      case 'R': // 현재 모델 카운트만 리셋
        resetCountsFor(ACTIVE_MODEL);
        updateLCD();
        CMD_SERIAL.println("카운트 리셋(현재 모델)");
        break;

      default:
        CMD_SERIAL.println("알 수 없는 명령");
        break;
    }
    lastSymbol = ch;
    lastCmdMs  = now;
  }
}