import sys
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QWidget, QPushButton
import cv2
import HelmetDetector_v1 as hd

class CameraApp(QWidget):
    def __init__(self, videopath):
        super().__init__()

        self.cap = cv2.VideoCapture(videopath)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.captured_image = None

        self.detector = hd.CamWrapper()  # HelmetDetector_v1에서 정의한 클래스 객체 생성
        self.detector.start()
        self.detector.bind()

        self.init_ui()
        self.blink_animation = QRect(0, 0, 0, 0)
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self.blink_timer_event)
        self.blink_timer.start(500)

    def init_ui(self):
        self.setWindowTitle('Webcam App')
        main_layout = QHBoxLayout(self)
        camera_frame = QFrame(self)
        camera_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        camera_frame.setLineWidth(2)
        main_layout.addWidget(camera_frame)
        capture_frame = QFrame(self)
        capture_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        capture_frame.setLineWidth(2)
        main_layout.addWidget(capture_frame)
        result_frame = QFrame(self)
        result_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        result_frame.setLineWidth(2)
        main_layout.addWidget(result_frame)
        camera_layout = QVBoxLayout(camera_frame)
        capture_layout = QVBoxLayout(capture_frame)
        result_layout = QVBoxLayout(result_frame)
        self.camera_label = QLabel(camera_frame)
        camera_layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)
        self.capture_label = QLabel(capture_frame)
        capture_layout.addWidget(self.capture_label, alignment=Qt.AlignCenter)
        self.result_label = QLabel(result_frame)
        result_layout.addWidget(self.result_label, alignment=Qt.AlignCenter)
        self.result_label.setText("안전모 착용 여부: 안전모 착용됨")
        capture_button = QPushButton('캡처', self)
        capture_button.clicked.connect(self.capture_image)
        capture_layout.addWidget(capture_button, alignment=Qt.AlignCenter)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.show()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            resized_frame = self.resize_frame(frame)

            # detect_human 메서드 적용
            detected_frame = self.detector.detect_human(resized_frame)

            q_image = self.convert_cv_frame_to_qimage(detected_frame)
            self.camera_label.setPixmap(QPixmap.fromImage(q_image))
            
            
            captured_image = self.detector.get_captured_image()
            if captured_image is not None:
                resized_captured_image = self.resize_frame(captured_image)
                q_captured_image = self.convert_cv_frame_to_qimage(resized_captured_image)
                self.capture_label.setPixmap(QPixmap.fromImage(q_captured_image))
    def blink_timer_event(self):
        self.capture_label.setGeometry(self.blink_animation)

    def resize_frame(self, frame):
        target_width = 640
        target_height = 480
        return cv2.resize(frame, (target_width, target_height))

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            self.captured_image = frame
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
        self.cap.release()
        self.detector.is_stop = True  # 프로그램 종료 시 detector 스레드를 종료하기 위해 is_stop 플래그 설정
        self.detector.join()  # 스레드 종료 대기
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    videopath = 'c:/Users/USER/Videos/clp/clptest2.mp4'
    window = CameraApp(videopath)
    sys.exit(app.exec_())
