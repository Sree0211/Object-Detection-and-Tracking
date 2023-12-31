from PyQt5.QtCore import QThread, pyqtSignal

import numpy as np
import cv2
import time

class VideoTracking(QThread):

    maskPixmap = pyqtSignal(np.ndarray)
    framePixmap = pyqtSignal(np.ndarray)


    def __init__(self):
        super().__init__()
        self._run_flag = True
    
    def run(self):
        cap = cv2.VideoCapture("/Users/sreenathswaminathan/Desktop/Uni-Docs/Autonomous fahren kurs/CV Project/Object-Detection-and-Tracking/pexels_videos_1171461 (1080p).mp4")
        object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=60)

        while self._run_flag:
            ret, frame = cap.read()
            # height, width, _ = frame.shape

            roi = frame[200:1000, 15:800]
            mask = object_detector.apply(roi)
            _, mask = cv2.threshold(mask, 253, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            if ret:
                self.framePixmap.emit(frame)
                self.maskPixmap.emit(mask)

            time.sleep(0.033)
            
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()