# 발 마우스 (Foot Mouse)

발 모션 인식을 통한 마우스 제어 프로그램입니다.

**2가지 버전을 제공합니다:**

## 📋 버전 선택

### v1: MediaPipe 버전
```bash
cd v1
pip install -r requirements.txt
python main.py
```

**특징:**
- ✓ 가볍고 빠름
- ✓ CPU 친화적
- ✗ 발만 보일 때 정확도 낮음 (상반신 필요)

**추천:** 빠른 성능이 필요할 때

---

### v2: YOLO 버전 (권장) ⭐
```bash
cd v2
pip install -r requirements.txt
python main.py
```

**특징:**
- ✓ 발만 보면 정확히 감지
- ✓ 높은 정확도
- ✓ 더 강건한 감지
- ✗ 더 무거움 (GPU 권장)

**추천:** 정확한 발 감지가 필요할 때 (대부분의 경우)

---

## 공통 조작 방법

### 🦶 오른발
- **좌우/위아래 이동** → 마우스 포인터 제어
- **위로 들었다 내리기** → 마우스 클릭

### ⌨️  키보드
- **'c'** → 현재 오른발 위치를 기준점으로 보정
- **'q'** → 프로그램 종료

---

## 각 버전 상세 정보

- [v1 (MediaPipe) 상세](v1/README.md)
- [v2 (YOLO) 상세](v2/README.md)

---

## 설정 팁

### 마우스가 너무 빠를 때
`config.py`에서:
```python
MOUSE_SCALE = 1.0  # 기본값: 2.0
```

### 마우스가 떨릴 때
```python
SMOOTHING_FACTOR = 0.8  # 기본값: 0.6
```

### 클릭이 자주 발동할 때
```python
CLICK_THRESHOLD = 0.1  # 기본값: 0.05
```

---

## 트러블슈팅

**발이 잘 안 보일 때**
- v1 → v2로 변경
- 조명 밝기 확인
- 카메라 위치 조정

**성능이 느릴 때**
- v1 사용
- GPU 설치 (v2용)
- 모델 크기 축소

---

## 요구사항

- Python 3.7+
- 웹캠
- 충분한 조명

각 버전의 `requirements.txt`를 참고하세요.
