#include <SoftwareSerial.h>

SoftwareSerial mySerial(2, 3); // RX = D2, TX = D3

void setup() {
  Serial.begin(115200);      // 디버깅용 (USB)
  mySerial.begin(115200);    // 소프트웨어 시리얼
  Serial.println("Arduino ready.");
}

void loop() {
  if (mySerial.available()) {
    char c = mySerial.read();
    mySerial.write(c);         // 받은 문자를 그대로 다시 전송 (에코)
    Serial.print("Echoed: ");
    Serial.println(c);
  }
}
