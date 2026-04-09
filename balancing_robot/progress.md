# 밸런싱 로봇 제작 진행상황

마지막 업데이트: 2026-04-09 (각도 계산 완료)

## 완료된 단계

### ✅ 1단계: 개발환경 설정
- STM32CubeIDE 설치 완료
- ST-LINK V2 드라이버 설치 완료
- 프로젝트 생성 완료 (balancing_robot)
- 프로젝트 위치: `C:\Users\PC\Desktop\balancing_robot\`

### ✅ CubeMX 핀 설정 완료
- SYS: Serial Wire (디버그)
- RCC: HSE Crystal/Ceramic Resonator → 72MHz
- I2C1: PB6(SCL), PB7(SDA) → MPU-6050용
- USART1: PA9(TX), PA10(RX), Asynchronous → HC-06용
- TIM2: CH1(PA0), CH2(PA1) PWM → 모터용, Prescaler=71, Period=999
- GPIO Output: PB0, PB1, PB10, PB11, PB12 → 모터 방향 제어용

### ✅ 보드 테스트
- 보드 2개 중 1개 정상, 1개 고장 확인
- 정상 보드: ST-LINK 연결 후 F11 디버그 성공 (HAL_Init()에서 정지 확인)
- 고장 보드: "No device found on target" 오류

---

### ✅ 2단계: MPU-6050 배선 완료
- GY-521 VCC → Blue Pill 3.3V
- GY-521 GND → 브레드보드 GND 행
- GY-521 SCL → Blue Pill PB6
- GY-521 SDA → Blue Pill PB7
- GY-521 AD0 → 브레드보드 GND 행
- Blue Pill GND → 브레드보드 GND 행

전원: ST-LINK V2 3.3V → Blue Pill V3 (USB/배터리 없이 사용 중)
- 마이크로 USB 케이블 없음
- 멀티미터 없음 (LM2596 전압 조정 불가 → 배터리 전원 연결 보류)

### ✅ 3단계: MPU-6050 통신 코드 작성 및 테스트 완료
- `Core/Inc/mpu6050.h`, `Core/Src/mpu6050.c` 작성 완료
- `main.c`에서 MPU6050_Init + MPU6050_Read 호출
- 링커 플래그 `-u _printf_float` 추가
- 디버거 Live Expressions로 센서값 확인 완료
  - 정지 상태에서 accel_z ≈ 1.0g (정상)
  - 자이로 노이즈 ±1~3 deg/s (정상 범위)

---

### ✅ 4단계: 각도 계산 (상보 필터) 완료
- `Core/Inc/angle.h`, `Core/Src/angle.c` 작성 완료
- 상보필터 구현: 0.98 × 자이로 + 0.02 × 가속도계
- X/Y 두 축 각도 모두 계산 (`angle.x`, `angle.y`)
- `HAL_GetTick()` 기반 dt 계산
- 센서 축 부호 보정 완료 (실측으로 확인)
- Live Expressions에서 실시간 각도 확인 완료

---

## 남은 단계

- [ ] 5단계: TB6612FNG + 모터 배선
- [ ] 6단계: 모터 제어 코드
- [ ] 7단계: PID 제어 통합
- [ ] 8단계: 튜닝
- [ ] 멀티미터 구하면: LM2596 출력 5V 조정 후 배터리 전원으로 전환
