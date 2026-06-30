import math
from collections import deque
from config import *

class MotionProcessor:
    def __init__(self):
        # 이전 발 위치 저장 (스무딩용)
        self.prev_x = None
        self.prev_y = None

        # 발 높이 히스토리 (클릭 감지용)
        self.height_history = deque(maxlen=10)

        # 초기 보정값 (선택적)
        self.calibration_offset = (0, 0)

    def process_motion(self, feet_data):
        """
        발 움직임 처리 및 마우스 좌표로 변환
        반환: {'mouse_x': x, 'mouse_y': y, 'foot_height': height, 'should_click': bool}
        """
        result = {
            'mouse_x': None,
            'mouse_y': None,
            'foot_height': None,
            'should_click': False,
            'foot_detected': False
        }

        # 발 감지 실패 시
        if not feet_data['detected']:
            return result

        result['foot_detected'] = True

        # 발 선택 (설정에 따라)
        if FOOT_MODE == 'right':
            foot_pos = feet_data['right_foot']
        elif FOOT_MODE == 'left':
            foot_pos = feet_data['left_foot']
        else:  # 'auto'
            foot_pos = feet_data['right_foot'] or feet_data['left_foot']

        if not foot_pos:
            return result

        raw_x, raw_y = foot_pos

        # 스무딩 적용
        if self.prev_x is None:
            smooth_x = raw_x
            smooth_y = raw_y
        else:
            smooth_x = self.prev_x * SMOOTHING_FACTOR + raw_x * (1 - SMOOTHING_FACTOR)
            smooth_y = self.prev_y * SMOOTHING_FACTOR + raw_y * (1 - SMOOTHING_FACTOR)

        self.prev_x = smooth_x
        self.prev_y = smooth_y

        # 카메라 좌표 → 화면 좌표 변환
        # 카메라에서 발 위치 (0~CAMERA_WIDTH) → 화면 좌표 (0~SCREEN_WIDTH)
        mouse_x = int((smooth_x / CAMERA_WIDTH) * SCREEN_WIDTH * MOUSE_SCALE)
        mouse_y = int((smooth_y / CAMERA_HEIGHT) * SCREEN_HEIGHT * MOUSE_SCALE)

        # 화면 범위 제한
        mouse_x = max(0, min(mouse_x, SCREEN_WIDTH - 1))
        mouse_y = max(0, min(mouse_y, SCREEN_HEIGHT - 1))

        result['mouse_x'] = mouse_x
        result['mouse_y'] = mouse_y
        result['foot_height'] = smooth_y

        # 클릭 감지 (발 높이 급격한 변화)
        self.height_history.append(smooth_y)
        if len(self.height_history) >= 5:
            height_diff = self.height_history[-1] - self.height_history[0]
            if height_diff < -CLICK_THRESHOLD * CAMERA_HEIGHT:  # 발을 들었을 때
                result['should_click'] = True

        return result

    def calibrate(self, feet_data):
        """초기 보정 - 현재 발 위치를 (0, 0)으로 설정"""
        if feet_data['detected'] and feet_data['right_foot']:
            self.calibration_offset = feet_data['right_foot']

    def reset(self):
        """상태 초기화"""
        self.prev_x = None
        self.prev_y = None
        self.height_history.clear()
