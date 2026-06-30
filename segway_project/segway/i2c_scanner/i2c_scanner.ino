#include <Wire.h>

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(100000);  // 100kHz 저속 모드

  Serial.println(F("I2C 스캐너 시작..."));
  Serial.println(F("연결된 I2C 장치를 찾습니다."));

  int count = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    uint8_t err = Wire.endTransmission();

    if (err == 0) {
      Serial.print(F("장치 발견! 주소: 0x"));
      if (addr < 16) Serial.print(F("0"));
      Serial.println(addr, HEX);
      count++;
    }
  }

  if (count == 0) {
    Serial.println(F("장치 없음. 배선 다시 확인하세요."));
  } else {
    Serial.print(count);
    Serial.println(F("개 장치 발견."));
  }
}

void loop() {}
