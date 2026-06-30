import cv2
import numpy as np
from ultralytics import YOLO
from config import *

class YOLOPoseDetector:
    def __init__(self):
        # YOLOv8 포즈 감지 모델 로드
        print("YOLOv8 포즈 감지 모델 로드 중...")
        self.model = YOLO('yolov8m-pose.pt')  # m: 중간 크기, 빠르고 정확함

        # GPU 설정
        if USE_GPU:
            try:
                self.model.to(f'cuda:{GPU_DEVICE}')
                print(f"✓ GPU 사용: {self.model.device}")
            except Exception as e:
                print(f"⚠️  GPU 설정 실패, CPU 사용: {e}")
        else:
            print("✓ CPU 모드로 실행 중")

    def detect_feet(self, frame):
        """
        발 좌표 감지 (YOLO)
        반환: {'left_foot': (x, y), 'right_foot': (x, y), 'results': results}
        """
        results = self.model(frame, verbose=False)[0]

        feet_data = {
            'left_foot': None,
            'right_foot': None,
            'results': results,
            'detected': False
        }

        # 감지된 사람이 있는지 확인
        if results.keypoints is None or len(results.keypoints) == 0:
            return feet_data

        feet_data['detected'] = True

        # 첫 번째 감지된 사람의 키포인트 가져오기
        keypoints = results.keypoints[0].xy[0]  # (17, 2) - 17개 관절, 각 x, y 좌표

        # YOLO 포즈 키포인트 인덱스:
        # 15: 왼발, 16: 오른발
        left_foot_idx = 15
        right_foot_idx = 16

        left_foot = keypoints[left_foot_idx]
        right_foot = keypoints[right_foot_idx]

        # 감지 신뢰도 확인
        if left_foot[0] > 0 and left_foot[1] > 0:  # (0, 0)이 아니면 감지됨
            feet_data['left_foot'] = (left_foot[0], left_foot[1])

        if right_foot[0] > 0 and right_foot[1] > 0:
            feet_data['right_foot'] = (right_foot[0], right_foot[1])

        return feet_data

    def draw_pose(self, frame, results):
        """
        YOLO 포즈 스켈레톤 그리기
        """
        if results is None:
            return frame

        # YOLO가 자동으로 그려줌
        annotated_frame = results.plot()
        return annotated_frame

    def draw_feet_points(self, frame, feet_data):
        """발 위치 포인트 강조하기"""
        if feet_data['left_foot']:
            x, y = int(feet_data['left_foot'][0]), int(feet_data['left_foot'][1])
            cv2.circle(frame, (x, y), 12, (255, 0, 0), -1)  # 파란색 (왼발)
            cv2.putText(frame, 'L', (x-10, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if feet_data['right_foot']:
            x, y = int(feet_data['right_foot'][0]), int(feet_data['right_foot'][1])
            cv2.circle(frame, (x, y), 12, (0, 0, 255), -1)  # 빨간색 (오른발)
            cv2.putText(frame, 'R', (x-10, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return frame

    def release(self):
        pass
