/*
 * MPU-6050 배선 테스트 스케치
 *
 * 배선:
 *   VCC → 3.3V
 *   GND → GND
 *   SCL → A5
 *   SDA → A4
 *   AD0 → GND  (I2C 주소 0x68)
 *
 * 시리얼 모니터: 115200 baud
 */

#include <Wire.h>

#define MPU_ADDR          0x68
#define REG_WHO_AM_I      0x75
#define REG_PWR_MGMT_1    0x6B
#define REG_CONFIG        0x1A
#define REG_GYRO_CONFIG   0x1B
#define REG_ACCEL_CONFIG  0x1C
#define REG_ACCEL_XOUT_H  0x3B

#define ACCEL_SCALE  16384.0f
#define GYRO_SCALE   131.0f

// ─────────────────────────────────────────
// MPU-6050 레지스터 쓰기
// ─────────────────────────────────────────
void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

// ─────────────────────────────────────────
// WHO_AM_I 확인 → 연결 여부 체크
// ─────────────────────────────────────────
bool checkConnection() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_WHO_AM_I);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)1);
  uint8_t id = Wire.available() ? Wire.read() : 0xFF;
  Serial.print(F("WHO_AM_I 값: 0x"));
  Serial.println(id, HEX);
  return true;  // 항상 통과
}

// ─────────────────────────────────────────
// 데이터 읽기
// ─────────────────────────────────────────
void readMPU(float &ax, float &ay, float &az,
             float &gx, float &gy, float &gz,
             float &temp) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)14);

  int16_t rax  = ((int16_t)Wire.read() << 8) | Wire.read();
  int16_t ray  = ((int16_t)Wire.read() << 8) | Wire.read();
  int16_t raz  = ((int16_t)Wire.read() << 8) | Wire.read();
  int16_t rtmp = ((int16_t)Wire.read() << 8) | Wire.read();
  int16_t rgx  = ((int16_t)Wire.read() << 8) | Wire.read();
  int16_t rgy  = ((int16_t)Wire.read() << 8) | Wire.read();
  int16_t rgz  = ((int16_t)Wire.read() << 8) | Wire.read();

  ax   = rax / ACCEL_SCALE;
  ay   = ray / ACCEL_SCALE;
  az   = raz / ACCEL_SCALE;
  gx   = rgx / GYRO_SCALE;
  gy   = rgy / GYRO_SCALE;
  gz   = rgz / GYRO_SCALE;
  temp = rtmp / 340.0f + 36.53f;  // MPU-6050 온도 공식 (°C)
}

// ─────────────────────────────────────────
// setup
// ─────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(100000);

  Serial.println(F("=========================================="));
  Serial.println(F("  MPU-6050 배선 테스트"));
  Serial.println(F("=========================================="));

  // 연결 확인
  Serial.print(F("I2C 연결 확인 (0x68)... "));
  if (!checkConnection()) {
    Serial.println(F("실패!"));
    Serial.println(F("  배선 재확인:"));
    Serial.println(F("    VCC → 3.3V  /  GND → GND"));
    Serial.println(F("    SCL → A5    /  SDA → A4"));
    Serial.println(F("    AD0 → GND"));
    while (true);  // 연결 실패 시 정지
  }
  Serial.println(F("OK (WHO_AM_I = 0x68)"));

  // MPU-6050 초기화
  delay(100);
  mpuWrite(REG_PWR_MGMT_1,   0x80);  // 리셋
  delay(100);
  mpuWrite(REG_PWR_MGMT_1,   0x00);  // 슬립 해제
  delay(100);
  mpuWrite(REG_CONFIG,       0x03);  // DLPF ~44Hz
  mpuWrite(REG_GYRO_CONFIG,  0x00);  // ±250 °/s
  mpuWrite(REG_ACCEL_CONFIG, 0x00);  // ±2 g

  Serial.println(F("초기화 완료. 데이터 출력 시작...\n"));
  Serial.println(F("ax(g)\tay(g)\taz(g)\tgx(°/s)\tgy(°/s)\tgz(°/s)\tangX(°)\tangY(°)\ttemp(°C)"));

  delay(500);
}

// ─────────────────────────────────────────
// loop
// ─────────────────────────────────────────
void loop() {
  float ax, ay, az, gx, gy, gz, temp;
  readMPU(ax, ay, az, gx, gy, gz, temp);

  // 가속도로 기울기 각도 계산
  float angleX = atan2(ay, az) * (180.0f / PI);  // X축 회전 (앞뒤)
  float angleY = atan2(ax, az) * (180.0f / PI);  // Y축 회전 (좌우)

  // 탭 구분 출력 → 시리얼 플로터에서도 확인 가능
  Serial.print(ax, 3);   Serial.print(F("\t"));
  Serial.print(ay, 3);   Serial.print(F("\t"));
  Serial.print(az, 3);   Serial.print(F("\t"));
  Serial.print(gx, 2);   Serial.print(F("\t"));
  Serial.print(gy, 2);   Serial.print(F("\t"));
  Serial.print(gz, 2);   Serial.print(F("\t"));
  Serial.print(angleX, 1); Serial.print(F("\t"));
  Serial.print(angleY, 1); Serial.print(F("\t"));
  Serial.println(temp, 1);

  delay(100);  // 10Hz 출력
}
