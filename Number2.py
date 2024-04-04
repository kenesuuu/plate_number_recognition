import easyocr
import cv2
import os
from os import listdir
from pathlib import Path
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QWidget, QSpacerItem
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import numpy as np
import time
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtMultimedia import QCamera, QCameraInfo
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimedia import QMediaRecorder
from PyQt5.QtMultimedia import QCameraInfo
from PyQt5.QtMultimedia import QCameraViewfinderSettings
from PyQt5.QtMultimedia import QCameraImageCapture


n_plate_detector = cv2.CascadeClassifier("C:/Users/Caffiene/Documents/11PlateNumber/plate_number_recognition/model/haarcascade_russian_plate_number.xml")

class CameraThread(QThread):
    frame_available = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect license plates using the Haar cascade classifier
                detections = n_plate_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)

                for (x, y, w, h) in detections:
                    # Draw a rectangle around the license plate
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                # Convert the frame to RGB and emit it
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_available.emit(rgb_frame)

    def stop(self):
        self.running = False
        self.wait()
        self.cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window title
        self.setWindowTitle("License Plate Detector")

        # Create a widget to hold the layout
        widget = QWidget()
        self.setCentralWidget(widget)

        # Create a vertical layout
        layout = QVBoxLayout()

        # Create a QLabel to display the processed frame
        self.label_preview = QLabel(self)
        layout.addWidget(self.label_preview)

        # Create a button to start the camera
        self.start_button = QPushButton("Start Camera")
        layout.addWidget(self.start_button)

        # Connect the button to a function to start the camera
        self.start_button.clicked.connect(self.start_camera)

        # Set the layout to the widget
        widget.setLayout(layout)

        # Initialize camera thread
        self.camera_thread = CameraThread()
        self.camera_thread.frame_available.connect(self.update_frame)

    def start_camera(self):
        # Start the camera thread
        self.camera_thread.start()

    def update_frame(self, frame):
        # Convert the frame to QImage and display it in the label
        h, w, ch = frame.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        self.label_preview.setPixmap(QPixmap.fromImage(p))

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())