import cv2
import mediapipe as mp
from config import *

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=CONFIDENCE_THRESHOLD,
            min_tracking_confidence=CONFIDENCE_THRESHOLD
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_feet(self, frame):
        """
        발 좌표 감지
        반환: {'left_foot': (x, y), 'right_foot': (x, y), 'landmarks': landmarks}
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        feet_data = {
            'left_foot': None,
            'right_foot': None,
            'landmarks': results.pose_landmarks,
            'detected': results.pose_landmarks is not None
        }

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, c = frame.shape

            # 왼발: 31(발뒤꿈치), 29(발가락)
            # 오른발: 32(발뒤꿈치), 30(발가락)
            left_foot_x = (landmarks[31].x + landmarks[29].x) / 2 * w
            left_foot_y = (landmarks[31].y + landmarks[29].y) / 2 * h
            feet_data['left_foot'] = (left_foot_x, left_foot_y)

            right_foot_x = (landmarks[32].x + landmarks[30].x) / 2 * w
            right_foot_y = (landmarks[32].y + landmarks[30].y) / 2 * h
            feet_data['right_foot'] = (right_foot_x, right_foot_y)

        return feet_data

    def draw_pose(self, frame, landmarks):
        """포즈 시각화 (디버그용)"""
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        return frame

    def draw_feet_points(self, frame, feet_data):
        """발 위치 포인트 그리기"""
        if feet_data['left_foot']:
            x, y = int(feet_data['left_foot'][0]), int(feet_data['left_foot'][1])
            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
            cv2.putText(frame, 'L', (x-10, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if feet_data['right_foot']:
            x, y = int(feet_data['right_foot'][0]), int(feet_data['right_foot'][1])
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(frame, 'R', (x-10, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def release(self):
        self.pose.close()
