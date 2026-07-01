import mujoco
import mujoco.viewer
import numpy as np
import threading
import sys
import tty
import termios

XML_PATH = "tron1-mujoco-sim/robot-description/pointfoot/PF_TRON1A/xml/robot.xml"

# 3개 값만 조작 (좌우 자동 미러링)
abad = 0.0
hip  = 0.0
knee = 0.0

KP      = 60.0
KD      =  3.0
STEP    =  0.05
KP_STEP =  5.0
KD_STEP =  0.5

CONTROLS = """
========== 터미널 조작 (엔터 없이) ==========
  i / k  :  hip  +/-     (앞뒤 기울기)
  j / l  :  knee +/-     (무릎 굽힘)
  u / o  :  abad +/-     (좌우 벌림)
  ] / [  :  KP   +/-
  ' / ;  :  KD   +/-
  r      :  리셋
  p      :  현재값 출력
  q      :  종료
=============================================
"""

def make_target(abad, hip, knee):
    # L/R 축이 반대이므로 R쪽은 부호 반전
    return np.array([abad,  hip,  knee,
                    -abad, -hip, -knee])

def print_state(params, gains):
    print(f"\r  abad={params[0]:+.2f}  hip={params[1]:+.2f}  knee={params[2]:+.2f}"
          f"  KP={gains[0]:.0f}  KD={gains[1]:.1f}  ", end='', flush=True)

def input_thread(params, gains, stop_flag, fd, old_term):
    try:
        tty.setraw(fd)
        while not stop_flag[0]:
            ch = sys.stdin.read(1)
            if   ch == 'i': params[1] += STEP
            elif ch == 'k': params[1] -= STEP
            elif ch == 'j': params[2] += STEP
            elif ch == 'l': params[2] -= STEP
            elif ch == 'u': params[0] += STEP
            elif ch == 'o': params[0] -= STEP
            elif ch == ']': gains[0] += KP_STEP
            elif ch == '[': gains[0] = max(0.0, gains[0] - KP_STEP)
            elif ch == "'": gains[1] += KD_STEP
            elif ch == ';': gains[1] = max(0.0, gains[1] - KD_STEP)
            elif ch == 'r':
                params[0] = params[1] = params[2] = 0.0
                print("\r[리셋]")
            elif ch == 'p':
                tq = make_target(*params)
                print(f"\r  TARGET_Q = {np.round(tq, 3).tolist()}")
                print(f"  KP={gains[0]}  KD={gains[1]}")
                continue
            elif ch == 'q': stop_flag[0] = True; break
            else: continue
            print_state(params, gains)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)

def run():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    params    = [abad, hip, knee]
    gains     = [KP, KD]
    stop_flag = [False]
    fd        = sys.stdin.fileno()
    old_term  = termios.tcgetattr(fd)

    print(CONTROLS)
    print_state(params, gains)

    t = threading.Thread(target=input_thread, args=(params, gains, stop_flag, fd, old_term), daemon=True)
    t.start()

    steps_per_frame = max(1, int(round(1.0 / 60.0 / model.opt.timestep)))

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.distance  = 3.0
        v.cam.elevation = -20

        while v.is_running() and not stop_flag[0]:
            target_q = make_target(*params)
            for _ in range(steps_per_frame):
                q  = data.qpos[7:]
                dq = data.qvel[6:]
                data.ctrl[:] = gains[0] * (target_q - q) + gains[1] * (0.0 - dq)
                mujoco.mj_step(model, data)
            v.sync()

    stop_flag[0] = True
    termios.tcsetattr(fd, termios.TCSADRAIN, old_term)

    tq = make_target(*params)
    print(f"\n\n[종료]  abad={params[0]:.3f}  hip={params[1]:.3f}  knee={params[2]:.3f}")
    print(f"  TARGET_Q = {np.round(tq, 3).tolist()}")
    print(f"  KP={gains[0]}  KD={gains[1]}")
    print(f"  base 높이: {data.qpos[2]:.3f} m")

if __name__ == "__main__":
    run()
