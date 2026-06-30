#!/usr/bin/env python3
"""GPU 및 YOLO 설정 테스트"""

import torch
from ultralytics import YOLO
from config import *

print("=" * 60)
print("🔧 발 마우스 v2 - GPU & YOLO 테스트")
print("=" * 60)

# 1. PyTorch CUDA 확인
print("\n1️⃣  PyTorch CUDA 정보:")
print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU 개수: {torch.cuda.device_count()}")
    print(f"   현재 GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"   메모리: {props.total_memory / 1e9:.1f}GB")
    print(f"   CUDA 버전: {torch.version.cuda}")
else:
    print("   ⚠️  GPU 미지원 - CPU만 사용 가능")

# 2. 설정 확인
print("\n2️⃣  발 마우스 v2 설정:")
print(f"   GPU 사용: {USE_GPU}")
print(f"   GPU 장치: {GPU_DEVICE}")
print(f"   카메라 해상도: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

# 3. YOLO 모델 로드 테스트
print("\n3️⃣  YOLOv8 모델 로드 테스트:")
try:
    print("   모델 로드 중...")
    model = YOLO('yolov8m-pose.pt')

    if USE_GPU and torch.cuda.is_available():
        print(f"   GPU로 이동 중...")
        model.to(f'cuda:{GPU_DEVICE}')
        print(f"   ✓ GPU 모드: {model.device}")
    else:
        print(f"   CPU 모드로 실행")

    print(f"   ✓ YOLOv8m-pose 모델 로드 성공")
except Exception as e:
    print(f"   ✗ 오류: {e}")
    exit(1)

# 4. 추론 속도 테스트
print("\n4️⃣  추론 속도 테스트:")
import numpy as np
import time

dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

# 워밍업
for _ in range(3):
    _ = model(dummy_frame, verbose=False)

# 실제 측정
times = []
for _ in range(10):
    start = time.time()
    _ = model(dummy_frame, verbose=False)
    times.append(time.time() - start)

avg_time = np.mean(times)
fps = 1.0 / avg_time

print(f"   평균 추론 시간: {avg_time*1000:.1f}ms")
print(f"   예상 FPS: {fps:.1f}")

if fps >= 15:
    print(f"   ✓ 충분한 성능")
else:
    print(f"   ⚠️  성능이 낮습니다. GPU 활용을 확인하세요.")

print("\n" + "=" * 60)
print("✅ 테스트 완료!")
print("=" * 60)
