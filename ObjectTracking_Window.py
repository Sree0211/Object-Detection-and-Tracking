import sys
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, pyqtSlot
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel, QPushButton, QProgressBar, QToolBar, QMainWindow
import numpy as np
import cv2
from ObjectTracking_Counting import VideoTracking

class VideoPanel(QWidget):
    def __init__(self, parent, label_text, geometry):
        super().__init__(parent)
        self.setGeometry(*geometry)

        self.image_label = QLabel(self)
        self.image_label.resize(700, 600)

        self.textLabel = QLabel(label_text, self)

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        self.setLayout(vbox)

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
        p = convert_to_Qt_format.scaled(700, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

class AppWindow(QMainWindow):
    def __init__(self, video_path):
        super().__init__()

        self.setWindowTitle("Real-time Object Tracking")
        self.setGeometry(100, 100, 1200, 1000)

        # Create Toolbar
        self.toolbar = self.addToolBar("Tools")
        self.toolbar.addAction("Action 1")
        self.toolbar.addAction("Action 2")

        # Create Video Panels
        self.video_panel_original = VideoPanel(self, "Object Detection", (20, 50, 700, 600))
        self.video_panel_masked = VideoPanel(self, "Masked Video for our ROI", (750, 50, 700, 600))

        # Create Pause/Resume Button
        self.pause_button = QPushButton('Pause', self)
        self.pause_button.setGeometry(100, 700, 120, 40)
        self.pause_button.clicked.connect(self.pause_resume_video)

        # Create Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(250, 700, 500, 40)

        # Activate timer
        self.timer = QTimer(self)
        
        # Create Video Thread
        self.video_thread = VideoTracking(video_path)
        self.video_thread.framePixmap.connect(self.video_panel_original.update_frame_image)
        self.video_thread.maskPixmap.connect(self.video_panel_masked.update_mask_image)
        self.video_thread.video_length_signal.connect(self.setup_progress_bar)
        self.video_thread.curr_signal.connect(self.update_progress_bar)
        self.video_thread.start()

        self.show()

    def pause_resume_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.pause_button.setText('Resume')
            self.video_thread.pause()
        else:
            self.timer.start(1000 / self.video_thread.frame_rate)
            self.pause_button.setText('Pause')
            self.video_thread.resume()

    def setup_progress_bar(self, total_frames):
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.total_frames = total_frames

    def update_progress_bar(self, current_frame):
        progress = int((current_frame / self.total_frames) * 100)
        if current_frame == self.total_frames:
            self.timer.stop()
        self.progress_bar.setValue(progress)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    video_path = "/Users/sreenathswaminathan/Desktop/Uni-Docs/Autonomous fahren kurs/CV Project/Object-Detection-and-Tracking/pexels_videos_1171461 (1080p).mp4"
    window = AppWindow(video_path)
    sys.exit(app.exec_())
