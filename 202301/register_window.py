import tkinter as tk
import json
from PIL import Image, ImageTk
from tkinter import messagebox
import cv2
import datetime

def open_register_window(root):#회원가입
    register_window = tk.Toplevel(root)
    register_window.title("회원가입")

    label_username = tk.Label(register_window, text="사용자명")
    label_username.pack()

    entry_username = tk.Entry(register_window, width=30)
    entry_username.pack()

    label_password = tk.Label(register_window, text="비밀번호")
    label_password.pack()

    entry_password = tk.Entry(register_window, show="*", width=30)
    entry_password.pack()

    label_confirm_password = tk.Label(register_window, text="비밀번호 확인")
    label_confirm_password.pack()

    entry_confirm_password = tk.Entry(register_window, show="*", width=30)
    entry_confirm_password.pack()

    button_register = tk.Button(register_window, text="회원가입", command=lambda: register(register_window, entry_username.get(), entry_password.get(), entry_confirm_password.get(),root))
    button_register.pack()
    

def register(register_window, username, password, confirm_password,root):#회원가입 확인
    if password == confirm_password:
        user_data = {"username": username, "password": password}
        label_result = tk.Label(root)
        with open("user_data.json", "w") as file:
            json.dump(user_data, file)
        label_result.config(text="회원가입 성공", fg="green")
        register_window.destroy()
    else:
        label_result.config(text="비밀번호가 일치하지 않습니다", fg="red")
