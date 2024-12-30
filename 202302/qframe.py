import sys
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QWidget, QPushButton
import cv2
import numpy as np
import HelmetDetector_v1 as hd


class CameraApp(QWidget):
    def __init__(self, videopath):
        super().__init__()

        # 웹 카메라 캡쳐용 변수
        self.cap = cv2.VideoCapture(videopath)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 캡쳐된 이미지 저장용 변수
        self.captured_image = None

        # UI 초기화
        self.init_ui()

        # 테두리 깜빡임 효과 초기화
        self.blink_animation = QRect(0, 0, 0, 0)
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self.blink_timer_event)
        self.blink_timer.start(500)  # 0.5초 간격으로 깜빡임

    def init_ui(self):
        self.setWindowTitle('Webcam App')

        # 전체 레이아웃 생성
        main_layout = QHBoxLayout(self)

        # 왼쪽 프레임 (카메라 화면)
        camera_frame = QFrame(self)
        camera_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        camera_frame.setLineWidth(2)
        main_layout.addWidget(camera_frame)

        # 중앙 프레임 (캡쳐된 이미지)
        capture_frame = QFrame(self)
        capture_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        capture_frame.setLineWidth(2)
        main_layout.addWidget(capture_frame)


        # 각 프레임의 레이아웃 생성
        camera_layout = QVBoxLayout(camera_frame)
        capture_layout = QVBoxLayout(capture_frame)
        

        # 비디오 출력용 라벨
        self.camera_label = QLabel(camera_frame)
        camera_layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)

        # 캡쳐된 이미지 출력용 라벨
        self.capture_label = QLabel(capture_frame)
        capture_layout.addWidget(self.capture_label, alignment=Qt.AlignCenter)

        # 캡쳐 버튼
        capture_button = QPushButton('캡처', self)
        capture_button.clicked.connect(self.capture_image)
        capture_layout.addWidget(capture_button, alignment=Qt.AlignCenter)

        # 타이머 시작
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.show()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 웹캠 프레임 크기 조절
            resized_frame = self.resize_frame(frame)
            # 웹캠 프레임에 테두리 적용
            q_image = self.convert_cv_frame_to_qimage(resized_frame)
            self.camera_label.setPixmap(QPixmap.fromImage(q_image))
            if self.captured_image is not None:
                resized_captured_image = self.resize_frame(self.captured_image)
                q_captured_image = self.convert_cv_frame_to_qimage(resized_captured_image)
                self.capture_label.setPixmap(QPixmap.fromImage(q_captured_image))    

    def blink_timer_event(self):
        # 테두리 깜빡임 효과 적용
        self.capture_label.setGeometry(self.blink_animation)

    def resize_frame(self, frame):
        # 원하는 크기로 프레임 조절
        target_width = 640
        target_height = 480
        return cv2.resize(frame, (target_width, target_height))

    def capture_image(self):
        # 현재 화면 캡처
        ret, frame = self.cap.read()
        if ret:
            self.captured_image = frame
            # 캡쳐된 이미지를 라벨에 표시
            q_image = self.convert_cv_frame_to_qimage(frame)
            self.capture_label.setPixmap(QPixmap.fromImage(q_image))
            print('이미지가 성공적으로 캡처되었습니다.')

    def convert_cv_frame_to_qimage(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return q_image

    def closeEvent(self, event):
        # 어플리케이션 종료 시 정리 작업 수행
        self.cap.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    videopath='c:/Users/USER/Videos/clp/clptest2.mp4'
    window = CameraApp(videopath)
    sys.exit(app.exec_())
