# 발 마우스 (Foot Mouse)

MediaPipe를 사용한 발 모션 인식 마우스 제어 프로그램입니다.

## 요구사항

- Python 3.7+
- 웹캠
- 충분한 조명

## 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

```bash
python main.py
```

### 조작 방법

#### 🦶 오른발 조작
- **마우스 포인터 이동**: 오른발을 좌우/위아래로 움직임
- **마우스 클릭**: 오른발을 위로 들었다 내리기

#### ⌨️  키보드 조작
- **'c' 키**: 보정 (현재 오른발 위치를 기준점으로 설정)
- **'q' 키**: 프로그램 종료

## 설정

`config.py`에서 다음 항목을 조정할 수 있습니다:

- `MOUSE_SCALE`: 마우스 이동 속도 (기본값: 2.0)
- `SMOOTHING_FACTOR`: 움직임 부드러움 정도 (0.0~1.0, 높을수록 부드러움)
- `CLICK_THRESHOLD`: 클릭 감지 민감도
- `DEBUG_MODE`: 화면에 포인트 표시 여부

## 파일 구조

- `main.py`: 메인 실행 파일
- `pose_detector.py`: MediaPipe 발 감지
- `motion_processor.py`: 발 움직임 처리 및 좌표 변환
- `mouse_controller.py`: 마우스 제어
- `config.py`: 설정값

## 팁

1. 처음 실행할 때 카메라를 향해 발을 들고 서세요
2. 발 감지가 잘 안 되면 조명을 더 밝게 하세요
3. `MOUSE_SCALE` 값을 조정하여 감도를 맞추세요
4. 손으로 카메라를 가리지 마세요 (포즈 감지 방해)

## 문제 해결

**발이 감지되지 않습니다**
- 조명을 밝게 해주세요
- 카메라가 발을 제대로 보는지 확인하세요
- `config.py`의 `CONFIDENCE_THRESHOLD`를 낮춰보세요 (예: 0.3)

**마우스가 너무 민감합니다**
- `MOUSE_SCALE` 값을 줄이세요 (예: 1.0)
- `SMOOTHING_FACTOR`를 높이세요 (예: 0.8)

**클릭이 자주 발동합니다**
- `CLICK_THRESHOLD`를 높이세요 (예: 0.1)
