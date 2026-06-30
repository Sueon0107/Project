# 발 마우스 v2 (YOLO 버전)

YOLOv8을 사용한 발 모션 인식 마우스 제어 프로그램입니다.

## v1 vs v2 비교

| 항목 | v1 (MediaPipe) | v2 (YOLO) |
|------|---------------|----------|
| **발 감지** | 전신 필요 | 발만 보면 OK ✓ |
| **정확도** | 중간 | 높음 ✓ |
| **속도** | 빠름 ✓ | 중간 |
| **조명** | 밝아야 함 | 더 강건함 ✓ |
| **모델 크기** | 작음 ✓ | 중간 |

## 요구사항

- Python 3.8+
- 웹캠
- 충분한 조명 (v1보다 덜 민감)
- CUDA 가능 GPU (선택사항, CPU도 작동하지만 느림)

## 설치

```bash
pip install -r requirements.txt
```

첫 실행 시 YOLOv8 모델이 자동으로 다운로드됩니다 (약 50MB).

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
- `pose_detector_yolo.py`: YOLO 기반 발 감지
- `motion_processor.py`: 발 움직임 처리 및 좌표 변환
- `mouse_controller.py`: 마우스 제어
- `config.py`: 설정값

## 팁

1. 처음 실행할 때 발만 카메라에 보이게 해주세요
2. v1보다 더 강건하지만 여전히 조명이 중요합니다
3. CUDA가 있으면 GPU 가속으로 더 빠릅니다
4. 손으로 카메라를 가리지 마세요

## GPU 사용 (선택)

CUDA가 설치되어 있으면 자동으로 GPU를 사용합니다.
```bash
# GPU 사용 확인
python -c "from ultralytics import YOLO; m = YOLO('yolov8m-pose.pt'); print(m.device)"
```

## 문제 해결

**발이 감지되지 않습니다**
- 카메라가 발을 제대로 보는지 확인하세요
- 조명을 밝게 해주세요

**메모리 부족**
- `pose_detector_yolo.py`에서 `yolov8s-pose.pt` (더 작은 모델)로 변경하세요

**느림**
- GPU 가용성 확인
- 또는 더 작은 모델(s) 사용
