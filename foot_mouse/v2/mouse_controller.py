import pyautogui
from collections import deque

class MouseController:
    def __init__(self):
        # 연속 클릭 방지를 위한 히스토리
        self.click_history = deque(maxlen=20)
        # 클릭 중복 방지 임계값 (프레임 수)
        self.click_cooldown = 15

    def move_mouse(self, x, y):
        """마우스 이동"""
        try:
            pyautogui.moveTo(x, y, duration=0)
        except Exception as e:
            print(f"마우스 이동 실패: {e}")

    def click(self):
        """마우스 클릭"""
        # 클릭 쿨다운 확인
        if len(self.click_history) > 0:
            if sum(self.click_history) > 0:  # 최근에 클릭했으면 무시
                return

        try:
            pyautogui.click()
            self.click_history.append(1)
            print("클릭!")
        except Exception as e:
            print(f"마우스 클릭 실패: {e}")

    def update_click_history(self):
        """매 프레임마다 쿨다운 감소"""
        if len(self.click_history) > 0:
            self.click_history.append(0)

    def disable_safety(self):
        """pyautogui 안전 기능 비활성화 (선택사항)"""
        pyautogui.FAILSAFE = False  # 마우스 모서리 이동하면 정지 기능 끄기
