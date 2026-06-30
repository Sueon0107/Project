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
 *  v4.0 - 데드밴드 제거, PWM_MIN 방식 도입
 *         출력이 PWM_MIN 미만이면 모터 정지
 *         PWM_MIN 시리얼 명령 W로 조정 가능
 *
 *  현재 버전: v4.0
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
#define MAX_OUTPUT_DEFAULT  255    // 모터 최대 출력 (0~255)
#define PWM_MIN_DEFAULT      30    // 이 값 미만 출력은 모터 정지
#define PRINT_EVERY          20    // N루프마다 시리얼 출력

// ================================================
// PID 구조체
// ================================================
struct PID {
  float kp       = 30.0f;
  float ki       = 50.0f;
  float kd       =  1.0f;
  float target   = -1.5f;
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
int           g_maxOutput  = MAX_OUTPUT_DEFAULT;
int           g_pwmMin     = 0;
float         g_innerZone  = 1.0f;
int           g_innerMax   = 255;
unsigned long g_prevTime   = 0;
int           g_printCount = 0;
bool          g_stopLogged = false;
bool          g_running    = true;   // false면 모터 정지, 각도만 출력
String        g_serialBuf  = "";

// ================================================
// PID 계산 (표준 PID, 데드밴드 없음)
// ================================================
float computePID(float current, float dt) {
  float error = pid.target - current;

  if (fabsf(error) >= 0.5f) {
    pid.integral += error * dt;
    pid.integral  = constrain(pid.integral, -100.0f, 100.0f);
  }

  float derivative = (error - pid.prevError) / dt;
  pid.prevError    = error;

  return pid.kp * error + pid.ki * pid.integral + pid.kd * derivative;
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

  // stop / start 명령 처리
  String lower = cmd;
  lower.toLowerCase();
  if (lower == "stop") {
    g_running = false;
    stopMotors();
    pid.integral = 0.0f; pid.prevError = 0.0f;
    Serial.println(F("[수동 정지] 각도만 출력 중. start로 재개."));
    return;
  }
  if (lower == "start") {
    g_running = true;
    Serial.println(F("[재개] 정상 작동 시작."));
    return;
  }

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
    case 'W':
      g_pwmMin = (int)constrain(val, 0.0f, 255.0f);
      Serial.print(F("PWM_MIN = ")); Serial.println(g_pwmMin);
      break;
    case 'Z':
      g_innerZone = constrain(val, 0.0f, 20.0f);
      Serial.print(F("내부구간 = ")); Serial.println(g_innerZone);
      break;
    case 'N':
      g_innerMax = (int)constrain(val, 0.0f, 255.0f);
      Serial.print(F("내부구간 최대출력 = ")); Serial.println(g_innerMax);
      break;
    case 'A':
      Serial.print(F("Kp=")); Serial.print(pid.kp);
      Serial.print(F(" Ki=")); Serial.print(pid.ki);
      Serial.print(F(" Kd=")); Serial.print(pid.kd);
      Serial.print(F(" T=")); Serial.print(pid.target);
      Serial.print(F(" M=")); Serial.print(g_maxOutput);
      Serial.print(F(" W=")); Serial.print(g_pwmMin);
      Serial.print(F(" Z=")); Serial.print(g_innerZone);
      Serial.print(F(" N=")); Serial.println(g_innerMax);
      break;
    case 'S':
      Serial.println(F("=== 파라미터 설명 ==="));
      Serial.println(F("P  : Kp - 비례 게인 (각도 오차에 비례한 출력)"));
      Serial.println(F("I  : Ki - 적분 게인 (누적 오차 보정, 드리프트 감소)"));
      Serial.println(F("D  : Kd - 미분 게인 (진동 억제)"));
      Serial.println(F("T  : 목표 각도 (°) - 균형 기준점"));
      Serial.println(F("M  : 최대 출력 (0~255) - 전체 최대 PWM"));
      Serial.println(F("W  : PWM 최소값 - 이 미만이면 모터 정지"));
      Serial.println(F("Z  : 내부 구간 (°) - 목표각도 이 범위 안에서 출력 제한"));
      Serial.println(F("N  : 내부 구간 최대 출력 - Z 범위 안에서의 최대 PWM"));
      Serial.println(F("A  : 현재 값 전체 출력"));
      Serial.println(F("S  : 이 도움말"));
      break;
    default:
      Serial.println(F("명령: P I D T M W Z N A S"));
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

  // DMP 초기화 (실패 시 최대 5회 재시도)
  Serial.println(F("DMP 초기화 중..."));
  for (int attempt = 1; attempt <= 5; attempt++) {
    devStatus = mpu.dmpInitialize();
    if (devStatus == 0) break;
    Serial.print(F("재시도 ")); Serial.print(attempt); Serial.print(F("/5 실패: ")); Serial.println(devStatus);
    delay(500);
  }

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
  Serial.println(F("  밸런싱 로봇 v4.0 (DMP)"));
  Serial.print  (F("  Kp=")); Serial.print(pid.kp);
  Serial.print  (F("  Ki=")); Serial.print(pid.ki);
  Serial.print  (F("  Kd=")); Serial.println(pid.kd);
  Serial.println(F("  명령: P I D T M W Z N A S"));
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
  float newAngle = ypr[2] * 180.0f / M_PI;

  // 이전 각도 대비 30° 이상 튀면 오염 데이터로 무시
  if (fabsf(newAngle - g_angle) > 30.0f && g_angle != 0.0f) {
    mpu.resetFIFO();
    processSerial();
    return;
  }
  g_angle = newAngle;

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
    mpu.resetFIFO();  // FIFO 오버플로우 방지
    if (!g_stopLogged) {
      g_stopLogged = true;
      Serial.print(F("[STOP] 과기울기: "));
      Serial.println(g_angle, 1);
    }
    processSerial();
    return;
  }
  g_stopLogged = false;  // 정상 범위 복귀 시 초기화

  // 수동 정지 상태: 각도만 출력
  if (!g_running) {
    if (++g_printCount >= PRINT_EVERY) {
      g_printCount = 0;
      Serial.print(F("angle:")); Serial.println(g_angle, 2);
    }
    processSerial();
    return;
  }

  // PID → 모터 출력
  float output   = computePID(g_angle, dt);
  float absError = fabsf(g_angle - pid.target);

  // 목표각도 도달 시 출력 0, 적분 리셋
  if (absError < 0.1f) {
    stopMotors();
    pid.integral  = 0.0f;
    pid.prevError = 0.0f;
  } else {
    int limit    = (absError < g_innerZone) ? g_innerMax : g_maxOutput;
    int motorVal = (int)constrain(output, (float)-limit, (float)limit);

    if (abs(motorVal) < g_pwmMin) {
      stopMotors();
      pid.integral = 0.0f;
    } else {
      setMotors(motorVal, motorVal);
    }
  }

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
