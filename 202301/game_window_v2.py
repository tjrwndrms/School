import tkinter as tk
import json
from PIL import Image, ImageTk
from tkinter import messagebox
import cv2
import datetime
from analysis_window import open_analysis_window
import time,os
cap = cv2.VideoCapture(0)

out = None  # 전역 변수로 out 선언
out_frames = []


def start_game(root):#게임 화면 구성
    game_window = tk.Toplevel(root)
    game_window.title("게임 화면")

    label_video = tk.Label(game_window)
    label_video.pack()

    label_start_game = tk.Label(game_window, text="게임을 시작합니다", font=("Arial", 24, "bold"))
    label_start_game.pack(pady=10)

    def show_video(): # 웹캠 화면  매개 변수 추가
        _, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image)
        label_video.configure(image=photo)
        label_video.image = photo
        label_video.after(60, show_video)

    show_video()

    def stop_game():
        if cap is not None:
            cap.release()
        label_start_game.config(text="게임을 종료합니다")
        game_window.destroy()

    
    def start_recording():
        os.makedirs('video', exist_ok=True)
        os.makedirs('video3', exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"video/user_{current_time}.mp4"
        video_filename3= f"video3/user_{current_time}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fps = 30.0
        recording_time = 3  # 녹화할 시간 (초)
        frame_count = int(fps * recording_time)
        
        global out
        out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
        
        
        # 1초 대기
        time.sleep(1)
        
        # 3초 동안 녹화

        for _ in range(frame_count):
            _, frame = cap.read()
            out.write(frame)
            out_frames.append(frame)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            label_video.configure(image=photo)
            label_video.image = photo
            game_window.update()  # 창 업데이트
        
        out.release()
        repeat_video(video_filename, video_filename3, 3)
        open_analysis_window(root,video_filename3)  # 녹화 종료 후 영상 분석 창 열기
        game_window.destroy()


    button_logout = tk.Button(game_window, text="로그아웃", command=lambda: logout(game_window,root))
    button_logout.pack(pady=10)

    button_start_recording = tk.Button(game_window, text="녹화 시작", command=start_recording)
    button_start_recording.pack(pady=10)

    button_stop_game = tk.Button(game_window, text="게임 종료", command=stop_game)
    button_stop_game.pack(pady=10)

    root.withdraw()

def logout(window,root):#위치 애매
    window.destroy()
    root.deiconify()



def repeat_video(video_path, output_path, repeat_count):
    cap = cv2.VideoCapture(video_path)

    # 원본 영상의 속성 확인
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 영상 반복하여 저장
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = frame_count * repeat_count

    for _ in range(repeat_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 영상의 프레임 위치를 처음으로 초기화
        for _ in range(frame_count):
            ret, frame = cap.read()
            if ret:
                out.write(frame)

    # 리소스 해제
    cap.release()
    out.release()

    print(f"New video saved: {output_path} (Repeated {repeat_count} times)")



