import cv2
import tkinter as tk
from PIL import ImageTk, Image

def guide(root, video_path):
    guide_window = tk.Toplevel(root)
    guide_window.title("Guide")

    guide_root = tk.Frame(guide_window)
    guide_root.pack()

    # 영상을 표시할 레이블 위젯을 생성합니다.
    label1 = tk.Label(guide_root)
    label1.grid(row=0, column=0, padx=10, pady=10)

    label2 = tk.Label(guide_root)
    label2.grid(row=0, column=1, padx=10, pady=10)

    label3 = tk.Label(guide_root)
    label3.grid(row=0, column=2, padx=10, pady=10)

    label4 = tk.Label(guide_root)
    label4.grid(row=0, column=3, padx=10, pady=10)

    # 사용자 영상, 테이크백, 시선, 힘에 대한 설명을 추가합니다.
    user_label = tk.Label(guide_root, text="사용자 영상")
    user_label.grid(row=1, column=0)

    takeback_label = tk.Label(guide_root, text="테이크백")
    takeback_label.grid(row=1, column=1)

    gaze_label = tk.Label(guide_root, text="시선")
    gaze_label.grid(row=1, column=2)

    force_label = tk.Label(guide_root, text="힘")
    force_label.grid(row=1, column=3)

    def play_video(video_path, label):
        cap = cv2.VideoCapture(video_path)

        def update_frame():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (300, int(frame.shape[0] * 300 / frame.shape[1])))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=image)
                label.configure(image=photo)
                label.image = photo

            # 영상의 마지막 프레임에 도달하면 첫 번째 프레임으로 되돌아갑니다.
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # 33ms마다 프레임을 업데이트합니다. (대략 30fps)
            guide_root.after(33, update_frame)

        update_frame()

    # 영상 파일 경로를 설정합니다.
    user_video_path = video_path
    takeback_video_path = "video/b.mp4"
    gaze_video_path = "video/c.mp4"
    force_video_path = "video/d.mp4"

    # 영상을 재생합니다.
    play_video(user_video_path, label1)
    play_video(takeback_video_path, label2)
    play_video(gaze_video_path, label3)
    play_video(force_video_path, label4)
