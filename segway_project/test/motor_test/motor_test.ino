// L298N Motor Driver Test - Arduino Nano
// 배선: IN1=D7, IN2=D8, ENA=D9(PWM)
// PWM 100% 연속 구동 - 모듈 스위치로 정지

const int IN1 = 7;
const int IN2 = 8;
const int ENA = 9;

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);

  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 255);
}

void loop() {}
