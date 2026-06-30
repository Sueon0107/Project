/*
 * 모터 동작 테스트
 * 앞으로 2초 → 정지 1초 → 뒤로 2초 → 정지 1초 반복
 */

#define PIN_IN1  5
#define PIN_IN2  6
#define PIN_ENA  3
#define PIN_IN3  9
#define PIN_IN4  10
#define PIN_ENB  11

#define SPEED 150  // 속도 0~255 (너무 빠르면 줄이세요)

void setMotors(int left, int right) {
  // 왼쪽 모터
  if (left > 0) {
    digitalWrite(PIN_IN1, HIGH); digitalWrite(PIN_IN2, LOW);
  } else if (left < 0) {
    digitalWrite(PIN_IN1, LOW); digitalWrite(PIN_IN2, HIGH);
    left = -left;
  } else {
    digitalWrite(PIN_IN1, LOW); digitalWrite(PIN_IN2, LOW);
  }
  analogWrite(PIN_ENA, left);

  // 오른쪽 모터
  if (right > 0) {
    digitalWrite(PIN_IN3, HIGH); digitalWrite(PIN_IN4, LOW);
  } else if (right < 0) {
    digitalWrite(PIN_IN3, LOW); digitalWrite(PIN_IN4, HIGH);
    right = -right;
  } else {
    digitalWrite(PIN_IN3, LOW); digitalWrite(PIN_IN4, LOW);
  }
  analogWrite(PIN_ENB, right);
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_IN1, OUTPUT); pinMode(PIN_IN2, OUTPUT); pinMode(PIN_ENA, OUTPUT);
  pinMode(PIN_IN3, OUTPUT); pinMode(PIN_IN4, OUTPUT); pinMode(PIN_ENB, OUTPUT);

  Serial.println(F("모터 테스트 시작"));
  delay(2000);  // 준비 시간
}

void loop() {
  Serial.println(F("앞으로"));
  setMotors(SPEED, SPEED);
  delay(2000);

  Serial.println(F("정지"));
  setMotors(0, 0);
  delay(1000);

  Serial.println(F("뒤로"));
  setMotors(-SPEED, -SPEED);
  delay(2000);

  Serial.println(F("정지"));
  setMotors(0, 0);
  delay(1000);
}
