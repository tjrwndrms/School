import tkinter as tk
import json
from PIL import Image, ImageTk
from tkinter import messagebox
import cv2
import datetime
from game_window_v2 import start_game
from register_window import open_register_window
out = None  # 전역 변수로 out 선언
out_frames = []  # 녹화된 프레임을 저장할 배열

def login(): #로그인 확인
    username = entry_username.get()
    password = entry_password.get()

    with open("user_data.json", "r") as file:
        user_data = json.load(file)

    if user_data["username"] == username and user_data["password"] == password:
        label_result.config(text="로그인 성공", fg="green")
        start_game(root)
    else:
        label_result.config(text="로그인 실패", fg="red")

root = tk.Tk()
root.attributes('-fullscreen', True)
root.title("회원가입 및 로그인")

image_path = "a.png"
image = Image.open(image_path)
image = image.resize((1700, 1000), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(image)

label_bg = tk.Label(root, image=photo)
label_bg.place(x=0, y=0, relwidth=1, relheight=1)

label_username = tk.Label(root, text="사용자명", width=30)
label_username.place(relx=0.5, rely=0.4, anchor="center")

entry_username = tk.Entry(root, width=30)
entry_username.place(relx=0.5, rely=0.45, anchor="center")

label_password = tk.Label(root, text="비밀번호", width=30)
label_password.place(relx=0.5, rely=0.5, anchor="center")

entry_password = tk.Entry(root, show="*", width=30)
entry_password.place(relx=0.5, rely=0.55, anchor="center")

button_register = tk.Button(root, text="회원가입", command=lambda: open_register_window(root), width=20)

button_register.place(relx=0.5, rely=0.6, anchor="center")

button_login = tk.Button(root, text="로그인", command=login, width=20)
button_login.place(relx=0.5, rely=0.65, anchor="center")

label_result = tk.Label(root)
label_result.place(relx=0.5, rely=0.7, anchor="center")



root.mainloop()