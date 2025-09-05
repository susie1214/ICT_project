#include <Dynamixel2Arduino.h>
#define DXL_SERIAL   Serial1
const int DXL_DIR_PIN = -1;
const uint8_t motor_ids[] = {1, 2, 3};
const uint8_t num_motors = 3;
#define RELAY_PIN 8  // 릴레이(흡착기) 제어 핀
// ---- 포즈 정의 ----
const uint16_t pose_default[] = {186, 211, 254};         // 기본 위치
const uint16_t pose_pickup[]  = {233, 256, 287};         // 픽업(흡착) 위치
const uint16_t pose_spade[]   = {163, 320, 330};         // 스페이드1
const uint16_t pose_spade2[]  = {208, 384, 321};         // 스페이드2
const uint16_t pose_club[]    = {163, 320, 330};         // 클럽1
const uint16_t pose_club2[]   = {195, 306, 379};         // 클럽2
const uint16_t pose_heart[]   = {313, 190, 232};         // 하트1
const uint16_t pose_heart2[]  = {337, 205, 315};         // 하트2
const uint16_t pose_dia[]     = {313, 190, 232};         // 다이아1
const uint16_t pose_dia2[]    = {341, 274, 227};         // 다이아2
// 이름 → 포즈 매핑
struct Pose {
  const char* name;
  const uint16_t* p;
};
Pose poses[] = {
  {"default", pose_default},
  {"pickup",  pose_pickup},
  {"spade1",  pose_spade},
  {"spade2",  pose_spade2},
  {"club1",   pose_club},
  {"club2",   pose_club2},
  {"heart1",  pose_heart},
  {"heart2",  pose_heart2},
  {"dia1",    pose_dia},
  {"dia2",    pose_dia2},
};
const size_t POSE_COUNT = sizeof(poses)/sizeof(poses[0]);
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
// 소문자로 변환
String lower(const String& s){
  String t = s;
  t.toLowerCase();
  return t;
}
void moveTo(const uint16_t p0, const uint16_t p1, const uint16_t p2){
  dxl.setGoalPosition(motor_ids[0], p0);
  dxl.setGoalPosition(motor_ids[1], p1);
  dxl.setGoalPosition(motor_ids[2], p2);
  Serial.print("이동 → ");
  Serial.print(p0); Serial.print(", ");
  Serial.print(p1); Serial.print(", ");
  Serial.println(p2);
}
bool moveByName(const String& name){
  String key = lower(name);
  for(size_t i=0;i<POSE_COUNT;i++){
    if(key == lower(String(poses[i].name))){
      const uint16_t* p = poses[i].p;
      moveTo(p[0], p[1], p[2]);
      return true;
    }
  }
  return false;
}
void printHelp(){
  Serial.println("=== 명령 안내 ===");
  Serial.println("s                : 현재 좌표 출력");
  Serial.println("t                : 토크 ON");
  Serial.println("k                : 토크 OFF");
  Serial.println("o / f            : 흡착 ON / OFF");
  Serial.println("<x> <y> <z>      : 좌표로 이동 (예: 200 250 300)");
  Serial.print  ("포즈 이름 입력   : ");
  for(size_t i=0;i<POSE_COUNT;i++){
    Serial.print(poses[i].name);
    if(i < POSE_COUNT-1) Serial.print(", ");
  }
  Serial.println();
  Serial.println("==================");
}
void printPositions(){
  Serial.print("현재 좌표: ");
  for (int i = 0; i < num_motors; i++) {
    uint16_t pos = dxl.getPresentPosition(motor_ids[i]);
    Serial.print("M"); Serial.print(motor_ids[i]);
    Serial.print(":"); Serial.print(pos);
    if (i < num_motors - 1) Serial.print(", ");
  }
  Serial.println();
}
void setup() {
  Serial.begin(115200);
  dxl.begin(115200);
  dxl.setPortProtocolVersion(1.0);
  for (int i = 0; i < num_motors; i++) {
    dxl.torqueOff(motor_ids[i]);
    dxl.setOperatingMode(motor_ids[i], OP_POSITION);
    dxl.torqueOn(motor_ids[i]);
  }
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  printHelp();
}
void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if(input.length() == 0) return;
    // 단일 문자 명령
    if (input.length() == 1) {
      char c = input[0];
      if (c=='s' || c=='S') { printPositions(); return; }
      if (c=='t' || c=='T') {
        for (int i=0;i<num_motors;i++) dxl.torqueOn(motor_ids[i]);
        Serial.println("토크 ON!");
        return;
      }
      if (c=='k' || c=='K') {
        for (int i=0;i<num_motors;i++) dxl.torqueOff(motor_ids[i]);
        Serial.println("토크 OFF!");
        return;
      }
      if (c=='o' || c=='O') { digitalWrite(RELAY_PIN, HIGH); Serial.println("흡착 ON!"); return; }
      if (c=='f' || c=='F') { digitalWrite(RELAY_PIN, LOW);  Serial.println("흡착 OFF!"); return; }
      if (c=='h' || c=='H' || c=='?') { printHelp(); return; }
    }
    // 좌표 3개 숫자 시도
    int x, y, z;
    if (sscanf(input.c_str(), "%d %d %d", &x, &y, &z) == 3) {
      moveTo((uint16_t)x, (uint16_t)y, (uint16_t)z);
      return;
    }
    // 포즈 이름 시도
    if (moveByName(input)) return;
    // 인식 실패
    Serial.println(" 알 수 없는 명령. 'h' 입력으로 도움말을 보세요.");
  }
}





