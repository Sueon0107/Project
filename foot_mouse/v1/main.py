#!/usr/bin/env python3
import cv2
import sys
from pose_detector import PoseDetector
from motion_processor import MotionProcessor
from mouse_controller import MouseController
from config import *

def main():
    # 초기화
    detector = PoseDetector()
    processor = MotionProcessor()
    mouse = MouseController()

    # 카메라 설정
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    print("=" * 50)
    print("발 마우스 시작")
    print("=" * 50)
    print("조작:")
    print("  🦶 [오른발]")
    print("    - 좌우/위아래 이동 → 마우스 포인터 제어")
    print("    - 위로 들었다 내리기 → 마우스 클릭")
    print("  ")
    print("  ⌨️  [키보드]")
    print("    - 'c': 보정 (오른발 위치를 기준점으로 설정)")
    print("    - 'q': 종료")
    print("=" * 50)

    # 윈도우 미리 생성
    if DEBUG_MODE:
        cv2.namedWindow('발 마우스', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('발 마우스', 800, 600)
        print("영상 창을 생성했습니다. 영상이 나타나길 기다리는 중...")

    calibrated = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("카메라 읽기 실패")
                break

            # 좌우 반전 (거울 모드)
            frame = cv2.flip(frame, 1)

            # 발 감지
            feet_data = detector.detect_feet(frame)

            # 움직임 처리
            motion = processor.process_motion(feet_data)

            # 마우스 제어
            if motion['foot_detected'] and motion['mouse_x'] is not None:
                mouse.move_mouse(motion['mouse_x'], motion['mouse_y'])

                if motion['should_click']:
                    mouse.click()

            mouse.update_click_history()

            # 터미널에 좌표 출력 (디버그용)
            if motion['foot_detected']:
                print(f"\r발 위치: ({motion['mouse_x']}, {motion['mouse_y']})", end="", flush=True)

            # 화면 표시 (디버그 모드)
            if DEBUG_MODE:
                # 스켈레톤 그리기
                if feet_data['landmarks']:
                    frame = detector.draw_pose(frame, feet_data['landmarks'])

                # 발 포인트 그리기
                frame = detector.draw_feet_points(frame, feet_data)

                # 정보 표시
                if motion['foot_detected']:
                    info_text = f"Mouse: ({motion['mouse_x']}, {motion['mouse_y']})"
                    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

                    if not calibrated:
                        status = "상태: 보정 필요 (c 누르기)"
                        cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 165, 255), 2)
                    else:
                        status = "상태: 활성 중"
                        cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "발 감지 안 됨", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

                try:
                    cv2.imshow('발 마우스', frame)
                except Exception as e:
                    print(f"영상 표시 오류: {e}")

            # 키 입력 처리
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("종료합니다...")
                break
            elif key == ord('c'):
                processor.calibrate(feet_data)
                calibrated = True
                print("✓ 보정 완료!")

    except KeyboardInterrupt:
        print("\n사용자 중단")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        print("종료되었습니다.")

if __name__ == "__main__":
    main()
