import sys
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt,QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtWidgets import QWidget,QApplication, QVBoxLayout, QLabel, QPushButton, QProgressBar
import numpy as np
import cv2
from ObjectTracking_Counting import VideoTracking


class VideoPanel(QWidget):
    def __init__(self, parent, label_text, geometry):
        super().__init__(parent)
        self.setGeometry(*geometry)

        self.image_label = QLabel(self)
        self.image_label.resize(800, 600)

        self.textLabel = QLabel(label_text, self)

        self.show()

    @pyqtSlot(np.ndarray)
    def update_mask_image(self, mask_img):
        qt_img = self.convert_cv_qt(mask_img)
        self.image_label.setPixmap(qt_img)

    @pyqtSlot(np.ndarray)
    def update_frame_image(self, frame_img):
        qt_img = self.convert_cv_qt(frame_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Real-time Object Tracking")
        self.setGeometry(100, 100, 1400, 700)

        self.create_video_panels()

        # Creating Pause and Play buttons
        self.pause_button = QPushButton('Pause')
        self.pause_button.move(100,100)
        
        self.pause_button.clicked.connect(self.pause_video)
        
        self.timer = QTimer(self)

        vbox = QVBoxLayout()
        vbox.addWidget(self.pause_button)

        self.progress = QProgressBar()
        vbox.addWidget(self.progress)

        self.setLayout(vbox)

        self.show()

    def create_video_panels(self):
        self.video_panel_original = VideoPanel(self, "Object Detection", (0, 0, 800, 800))
        self.video_panel_masked = VideoPanel(self, "Masked Video", (800, 100, 500, 500))

        self.thread = VideoTracking()
        self.thread.framePixmap.connect(self.video_panel_original.update_frame_image)
        self.thread.maskPixmap.connect(self.video_panel_masked.update_mask_image)
        self.thread.start()

    def pause_video(self):
        # Resume the video thread again
        if self.timer.isActive():
            self.timer.stop()
            self.pause_button.setText('Resume')
        else:
            self.timer.timeout.connect()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept() 

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AppWindow()

    window.show()
    sys.exit(app.exec_())

