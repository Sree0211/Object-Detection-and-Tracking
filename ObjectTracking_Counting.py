from PyQt5.QtCore import QThread, pyqtSignal, QTimer

import numpy as np
import cv2
import time

class VideoTracking(QThread):

    # Required Signals to connect with the GUI
    # videoEnded = pyqtSignal()

    maskPixmap = pyqtSignal(np.ndarray)
    framePixmap = pyqtSignal(np.ndarray)
    video_length_signal = pyqtSignal(int)
    curr_signal = pyqtSignal(int)

    def __init__(self, video_path):
        super().__init__()
        self._run_flag = True
        self._paused = False
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.current_frame = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def run(self):
        object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60)

        while self._run_flag and self.current_frame < self.total_frames:
            if not self._paused:
                ret, frame = self.cap.read()
                if ret:
                    roi = frame[200:1000, 15:800]
                    mask = object_detector.apply(roi)
                    _, mask = cv2.threshold(mask, 253, 255, cv2.THRESH_BINARY)

                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    for cnt in contours:
                        
                        area = cv2.contourArea(cnt)
                        # Perform Non-Maximum Suppression (NMS)
                        cnt = self.non_max_suppression(cnt, 0.5)
                        if area > 100:

                            cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                            x, y, w, h = cv2.boundingRect(cnt)
                            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

                    self.maskPixmap.emit(mask)
                    self.framePixmap.emit(frame)
                    self.video_length_signal.emit(self.total_frames)
                    self.curr_signal.emit(self.current_frame)

                    self.current_frame += 1
                
                time.sleep(0.033)

        self.cap.release()

    def pause(self):
        self.timer.stop()
        self._paused = True

    def resume(self):
        self.timer.start(1000 / self.frame_rate)
        self._paused = False

    def stop(self):
        self._run_flag = False

    def update_frame(self):
        self.timer.stop()
        self.resume()

    def non_max_suppression(self,boxes, overlap_thresh):
        if len(boxes) == 0:
            return []

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(boxes[:, 4])

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        return boxes[pick]