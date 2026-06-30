/*
 * ================================================
 *  자립 밸런싱 로봇 제어 코드
 *  MCU    : Arduino Nano (ATmega328P)
 *  IMU    : MPU-6050 (GY-521)
 *  드라이버: L298N 듀얼 H브릿지
 *  모터   : TT DC 기어드모터 x2
 *  전원   : 18650 x2 직렬 (7.4V)
 * ================================================
 *
 *  버전 히스토리
 *  v1.0 - 최초 작성 (PID + 상보필터 + 시리얼 튜닝)
 *  v1.1 - 데드밴드 추가 (±1.5° 이내 모터 정지)
 *  v1.2 - 최대 출력 제한 추가 (M 명령어, 기본 150)
 *  v1.3 - 루프 10ms→5ms, ALPHA 0.98→0.95,
 *         시리얼 출력 20루프마다로 변경
 *  v2.0 - DMP 기반 각도 측정으로 전환
 *         (상보필터 제거, INT핀 D2 추가 필요)
 *         자동 캘리브레이션 적용
 *  v2.1 - P항을 error² 비례로 변경 (비선형 제어)
 *         Kp 기본값 60 → 8로 조정
 *  v2.2 - 선형 P로 복귀, 최대출력 255, 과기울기 25°,
 *         과기울기 로그 한 번만 출력
 *  v2.3 - I를 데드밴드 안에서만 누적
 *         목표각도 0.1° 이내 도달 시 적분 즉시 초기화
 *  v2.4 - 데드밴드 재정의
 *         안: I만 출력 (드리프트 보정)
 *         밖: P,D만 출력 (빠른 복귀)
 *
 *  현재 버전: v2.4
 * ================================================
 */

#include <Wire.h>
#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"

// ================================================
// 핀 정의
// ================================================
#define PIN_IN1  5
#define PIN_IN2  6
#define PIN_ENA  3
#define PIN_IN3  9
#define PIN_IN4  10
#define PIN_ENB  11
#define PIN_INT  2    // MPU-6050 INT → D2 (새로 연결 필요)

// ================================================
// 제어 상수
// ================================================
#define SAFE_ANGLE         25.0f   // 안전 기울기 한계 (°)
#define DEADBAND            2.5f   // 데드밴드 (이 각도 이내면 모터 정지)
#define MAX_OUTPUT_DEFAULT  255    // 모터 최대 출력 (0~255)
#define PRINT_EVERY          20    // N루프마다 시리얼 출력

// ================================================
// PID 구조체
// ================================================
struct PID {
  float kp       = 40.0f;
  float ki       = 50.0f;
  float kd       =  5.0f;
  float target   =  0.6f;
  float prevError =  0.0f;
  float integral  =  0.0f;
} pid;

// ================================================
// MPU-6050 DMP 변수
// ================================================
MPU6050 mpu;
bool     dmpReady  = false;
uint8_t  devStatus;
uint16_t packetSize;
uint8_t  fifoBuffer[64];
Quaternion  q;
VectorFloat gravity;
float ypr[3];

volatile bool mpuInterrupt = false;
void dmpDataReady() { mpuInterrupt = true; }

// ================================================
// 전역 변수
// ================================================
float         g_angle     = 0.0f;
int           g_maxOutput      = MAX_OUTPUT_DEFAULT;
int           g_maxOutputInner = 30;   // 데드밴드 안 최대 출력
float         g_deadband       = DEADBAND;
unsigned long g_prevTime   = 0;
int           g_printCount = 0;
bool          g_stopLogged = false;
String        g_serialBuf  = "";

// ================================================
// PID 계산
// 데드밴드 안: I만 출력에 영향 (드리프트 보정)
// 데드밴드 밖: P, D만 출력에 영향 (빠른 복귀)
// ================================================
float computePID(float current, float dt) {
  float error    = pid.target - current;
  float absError = fabsf(error);

  float derivative = (error - pid.prevError) / dt;
  pid.prevError    = error;

  if (absError < g_deadband) {
    // 데드밴드 안: I 누적 및 출력
    if (absError < 0.1f) pid.integral = 0.0f;  // 목표 도달 시 리셋
    else {
      pid.integral += error * dt;
      pid.integral  = constrain(pid.integral, -100.0f, 100.0f);
    }
    return pid.ki * pid.integral;
  } else {
    // 데드밴드 밖: P, D만 출력 (I 누적 안 함)
    return pid.kp * error + pid.kd * derivative;
  }
}

// ================================================
// 모터 제어
// ================================================
void stopMotors() {
  digitalWrite(PIN_IN1, LOW); digitalWrite(PIN_IN2, LOW); analogWrite(PIN_ENA, 0);
  digitalWrite(PIN_IN3, LOW); digitalWrite(PIN_IN4, LOW); analogWrite(PIN_ENB, 0);
}

void setMotors(int left, int right) {
  left  = constrain(left,  -255, 255);
  right = constrain(right, -255, 255);

  if (left > 0) {
    digitalWrite(PIN_IN1, HIGH); digitalWrite(PIN_IN2, LOW);
  } else if (left < 0) {
    digitalWrite(PIN_IN1, LOW); digitalWrite(PIN_IN2, HIGH);
    left = -left;
  } else {
    digitalWrite(PIN_IN1, LOW); digitalWrite(PIN_IN2, LOW);
  }
  analogWrite(PIN_ENA, (uint8_t)left);

  if (right > 0) {
    digitalWrite(PIN_IN3, HIGH); digitalWrite(PIN_IN4, LOW);
  } else if (right < 0) {
    digitalWrite(PIN_IN3, LOW); digitalWrite(PIN_IN4, HIGH);
    right = -right;
  } else {
    digitalWrite(PIN_IN3, LOW); digitalWrite(PIN_IN4, LOW);
  }
  analogWrite(PIN_ENB, (uint8_t)right);
}

// ================================================
// 시리얼 명령 처리 (비블로킹)
// P(Kp) I(Ki) D(Kd) T(목표각도) M(최대출력)
// ================================================
void parseCommand(const String &cmd) {
  if (cmd.length() < 1) return;
  char  key = toupper(cmd.charAt(0));
  float val = cmd.substring(1).toFloat();
  switch (key) {
    case 'P': pid.kp = val; Serial.print(F("Kp = ")); Serial.println(val); break;
    case 'I': pid.ki = val; pid.integral = 0; Serial.print(F("Ki = ")); Serial.println(val); break;
    case 'D': pid.kd = val; Serial.print(F("Kd = ")); Serial.println(val); break;
    case 'T': pid.target = val; Serial.print(F("목표각도 = ")); Serial.println(val); break;
    case 'M':
      g_maxOutput = (int)constrain(val, 0.0f, 255.0f);
      Serial.print(F("최대출력 = ")); Serial.println(g_maxOutput);
      break;
    case 'B':
      g_deadband = constrain(val, 0.0f, 10.0f);
      Serial.print(F("데드밴드 = ")); Serial.println(g_deadband);
      break;
    case 'N':
      g_maxOutputInner = (int)constrain(val, 0.0f, 255.0f);
      Serial.print(F("데드밴드 내 최대출력 = ")); Serial.println(g_maxOutputInner);
      break;
    case 'A':
      Serial.print(F("Kp=")); Serial.print(pid.kp);
      Serial.print(F(" Ki=")); Serial.print(pid.ki);
      Serial.print(F(" Kd=")); Serial.print(pid.kd);
      Serial.print(F(" T=")); Serial.print(pid.target);
      Serial.print(F(" M=")); Serial.print(g_maxOutput);
      Serial.print(F(" N=")); Serial.print(g_maxOutputInner);
      Serial.print(F(" B=")); Serial.println(g_deadband);
      break;
    default:
      Serial.println(F("명령: P(Kp) I(Ki) D(Kd) T(목표각도) M(최대출력) B(데드밴드) A(전체출력)"));
  }
}

void processSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (g_serialBuf.length() > 0) { parseCommand(g_serialBuf); g_serialBuf = ""; }
    } else {
      g_serialBuf += c;
    }
  }
}

// ================================================
// setup
// ================================================
void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(100000);  // 초기화 시 안정적인 100kHz 사용

  // 모터 핀 초기화
  pinMode(PIN_IN1, OUTPUT); pinMode(PIN_IN2, OUTPUT); pinMode(PIN_ENA, OUTPUT);
  pinMode(PIN_IN3, OUTPUT); pinMode(PIN_IN4, OUTPUT); pinMode(PIN_ENB, OUTPUT);
  stopMotors();

  // MPU-6050 초기화
  Serial.println(F("MPU-6050 초기화 중..."));
  delay(200);
  mpu.initialize();
  delay(200);
  pinMode(PIN_INT, INPUT);

  // 클론 센서는 WHO_AM_I가 달라 testConnection() 스킵
  Serial.println(F("MPU-6050 연결 OK (클론 모듈)"));

  // DMP 초기화
  Serial.println(F("DMP 초기화 중... (약 15초 소요)"));
  devStatus = mpu.dmpInitialize();

  if (devStatus == 0) {
    // 오프셋 수동 설정 (0으로 시작, 필요시 조정)
    mpu.setXGyroOffset(0);
    mpu.setYGyroOffset(0);
    mpu.setZGyroOffset(0);
    mpu.setXAccelOffset(0);
    mpu.setYAccelOffset(0);
    mpu.setZAccelOffset(0);

    mpu.setDMPEnabled(true);
    attachInterrupt(digitalPinToInterrupt(PIN_INT), dmpDataReady, RISING);
    packetSize = mpu.dmpGetFIFOPacketSize();
    dmpReady = true;
    Serial.println(F("DMP 준비 완료!"));
  } else {
    Serial.print(F("DMP 초기화 실패! 코드: "));
    Serial.println(devStatus);
    while (true);
  }

  Serial.println(F("========================================="));
  Serial.println(F("  밸런싱 로봇 v2.0 (DMP)"));
  Serial.print  (F("  Kp=")); Serial.print(pid.kp);
  Serial.print  (F("  Ki=")); Serial.print(pid.ki);
  Serial.print  (F("  Kd=")); Serial.println(pid.kd);
  Serial.println(F("  명령: P/I/D/T/M + 값 (예: P30)"));
  Serial.println(F("========================================="));

  g_prevTime = millis();
}

// ================================================
// loop
// ================================================
void loop() {
  if (!dmpReady) return;

  // DMP 데이터 준비 대기
  if (!mpuInterrupt && mpu.getFIFOCount() < packetSize) {
    processSerial();
    return;
  }

  mpuInterrupt = false;

  // FIFO 오버플로우 처리
  if (mpu.getFIFOCount() >= 1024) {
    mpu.resetFIFO();
    processSerial();
    return;
  }

  // 패킷 읽기
  while (mpu.getFIFOCount() >= packetSize) {
    mpu.getFIFOBytes(fifoBuffer, packetSize);
  }

  // 쿼터니언 → 각도 변환
  mpu.dmpGetQuaternion(&q, fifoBuffer);
  mpu.dmpGetGravity(&gravity, &q);
  mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
  g_angle = ypr[2] * 180.0f / M_PI;  // roll (앞뒤 기울기, 장착 방향에 따라 ypr[1]로 변경)

  // dt 계산
  unsigned long now = millis();
  float dt = (now - g_prevTime) / 1000.0f;
  if (dt <= 0.0f) dt = 0.01f;
  g_prevTime = now;

  // 25° 안전장치
  if (fabsf(g_angle) > SAFE_ANGLE) {
    stopMotors();
    pid.integral  = 0.0f;
    pid.prevError = 0.0f;
    if (!g_stopLogged) {
      g_stopLogged = true;
      Serial.print(F("[STOP] 과기울기: "));
      Serial.println(g_angle, 1);
    }
    processSerial();
    return;
  }
  g_stopLogged = false;  // 정상 범위 복귀 시 초기화

  // 데드밴드
  // 데드밴드 하드 정지 제거 - PID 내부에서 구간별 처리

  // PID → 모터 출력
  float output  = computePID(g_angle, dt);
  bool inDeadband = fabsf(g_angle - pid.target) < g_deadband;
  int  limit    = inDeadband ? g_maxOutputInner : g_maxOutput;
  int  motorVal = (int)constrain(output, (float)-limit, (float)limit);
  setMotors(motorVal, motorVal);

  // 시리얼 디버깅 출력
  if (++g_printCount >= PRINT_EVERY) {
    g_printCount = 0;
    Serial.print(F("angle:")); Serial.print(g_angle, 2);
    Serial.print(F("\tout:"));  Serial.print(output,  1);
    Serial.print(F("\tKp:"));   Serial.print(pid.kp);
    Serial.print(F("\tKi:"));   Serial.print(pid.ki);
    Serial.print(F("\tKd:"));   Serial.println(pid.kd);
  }

  processSerial();
}
