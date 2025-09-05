#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// LCD 설정 (주소: 0x27, 크기: 16x2)
LiquidCrystal_I2C lcd(0x27, 16, 2);

// 시리얼 포트 정의
#define CMD_SERIAL   Serial2   // TX/RX 포트 (맥북 ↔ OpenRB)
#define DEBUG_SERIAL Serial    // USB 디버깅용 시리얼

// 표시할 데이터
String pass_card = "S7";  // 예: S7, D10 등
int count_spade = 2;
int count_club = 0;
int count_diamond = 5;
int count_heart = 1;

void setup() {
  DEBUG_SERIAL.begin(115200);   // USB 디버깅 시리얼
  CMD_SERIAL.begin(115200);     // 맥북 ↔ OpenRB 시리얼

  // LCD 초기화
  lcd.init();
  lcd.backlight();

  // LCD 첫 줄: 패스카드 정보
  lcd.setCursor(0, 0);
  lcd.print("PASS CARD : ");
  lcd.print(pass_card);

  // LCD 둘째 줄: 문양별 개수
  lcd.setCursor(0, 1);
  lcd.print("S:");
  lcd.print(count_spade);
  lcd.print(" C:");
  lcd.print(count_club);
  lcd.print(" D:");
  lcd.print(count_diamond);
  lcd.print(" H:");
  lcd.print(count_heart);

  DEBUG_SERIAL.println("▶ 에코모드 시작, 종료하려면 USB로 아무 키나 입력");
}

void loop() {
  // 1. 맥북 ↔ OpenRB 시리얼 통신 에코 처리
  if (CMD_SERIAL.available()) {
    char received = CMD_SERIAL.read();

    // 에코 출력
    DEBUG_SERIAL.print("수신: ");
    DEBUG_SERIAL.println(received);

    CMD_SERIAL.write(received);  // 다시 맥북으로 에코

    DEBUG_SERIAL.print("송신 (에코): ");
    DEBUG_SERIAL.println(received);
  }

  // 2. USB 시리얼(Serial)로 종료 명령 감지
  if (DEBUG_SERIAL.available()) {
    char c = DEBUG_SERIAL.read();
    DEBUG_SERIAL.print("종료 명령 수신: ");
    DEBUG_SERIAL.println(c);

    // LCD 종료 표시
    lcd.clear();
    lcd.setCursor(1, 0);
    lcd.print("LCD 종료");

    while (1); // 무한 정지
  }
}
