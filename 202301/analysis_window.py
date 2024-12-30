import tkinter as tk
import json
from PIL import Image, ImageTk
from tkinter import messagebox
import cv2
import datetime
from get_grade import get_grade
from guide import guide
def open_analysis_window(root,video_path):#수정해서 미디어파이프 나오게
    grade=get_grade(video_path)
    open_grade_window(root,grade,video_path)
    

def open_grade_window(root,grade,video_path):
    grade_window = tk.Toplevel(root)
    grade_window.title("등급")

    label_grade = tk.Label(grade_window, text=f'당신의 등급은 {grade} 입니다!', font=("Arial", 24, "bold"))
    label_grade.pack(pady=10)

    def go_back():
        grade_window.destroy()
        root.deiconify()
        
    guide(root,video_path)
    

    button_go_back = tk.Button(grade_window, text="돌아가기", command=go_back)
    button_go_back.pack()
