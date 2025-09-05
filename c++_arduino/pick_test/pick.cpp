#define RELAY_PIN 8

void setup() {
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  Serial.begin(115200);
  Serial.println("흡착기 제어 시작: 't'=ON, 'f'=OFF");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();

    // 개행 문자 무시
    if (cmd == '\r' || cmd == '\n') {
      return;
    }

    if (cmd == 't') {
      digitalWrite(RELAY_PIN, HIGH);
      Serial.println("흡착기 ON");
    } 
    else if (cmd == 'f') {
      digitalWrite(RELAY_PIN, LOW);
      Serial.println("흡착기 OFF");
    } 
    else {
      Serial.print("알 수 없는 명령: ");
      Serial.println(cmd);
    }
  }
}
