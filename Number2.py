import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QWidget, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import numpy as np
import easyocr
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained Haar cascade classifier for license plate detection
n_plate_detector = cv2.CascadeClassifier("C:/Users/Caffiene/Documents/11PlateNumber/plate_number_recognition/model/haarcascade_russian_plate_number.xml")


class CameraThread(QThread):
    frame_available = pyqtSignal(np.ndarray)
    frame_skip = 3 # Process every 10th frame

    def __init__(self, video_source):
        super().__init__()
        self.cap = cv2.VideoCapture(video_source)
        self.frame_count = 0
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                if self.frame_count % self.frame_skip == 0:
                    self.frame_available.emit(frame)

    def stop(self):
        self.running = False
        self.wait()
        self.cap.release()

class OCRThread(QThread):
    result_signal = pyqtSignal(str)

    def __init__(self, plate_region):
        super().__init__()
        self.plate_region = plate_region
        self.reader = easyocr.Reader(['en'], gpu=True) # Initialize OCR reader with GPU support

    def run(self):
        plate_text = self.read_license_plate(self.plate_region)
        self.result_signal.emit(plate_text)

    def read_license_plate(self, plate_region):
        gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_plate, 150, 255, cv2.THRESH_BINARY_INV)
        plate_text = self.reader.readtext(thresh, detail=0, paragraph=False)
        if plate_text:
            return plate_text[0][-2] # Assuming the first recognized text is the license plate number
        else:
            return "Not Recognized"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("License Plate Detector")
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        self.label_preview = QLabel(self)
        layout.addWidget(self.label_preview)
        self.start_button = QPushButton("Start Camera")
        layout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start_camera)
        self.record_button = QPushButton("Record Details")
        layout.addWidget(self.record_button)
        self.record_button.clicked.connect(self.recorded_details) # Corrected method name
        widget.setLayout(layout)
        self.camera_thread = None
        self.recorded_details = []
        self.save_folder = "recorded_details"
        # Create the save folder if it doesn't exist
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def handle_ocr_result(self, plate_text):
            # Assuming this method is called with the plate text as an argument
            # You can now handle the OCR result, e.g., display it or store it
            print(f"Plate Text: {plate_text}")

    def recorded_details(self):
        if self.recorded_details:
            for i, (plate_text, plate_region) in enumerate(self.recorded_details):
                cv2.imwrite(f"{self.save_folder}/detected_plate_{i}.jpg", plate_region)
                print(f"Plate Number: {plate_text} - Details saved as detected_plate_{i}.jpg")

        # Create the save folder if it doesn't exist
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            

    def start_camera(self):
        video_source = 0 # Use 0 for default camera, you can change it to file path for video files
        self.camera_thread = CameraThread(video_source)
        self.camera_thread.frame_available.connect(self.update_frame)
        self.camera_thread.start()
        # Set a lower resolution for the video feed
        self.camera_thread.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera_thread.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def update_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = n_plate_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)

        for (x, y, w, h) in detections:
            plate_region = frame[y:y + h, x:x + w] # Extract license plate region
            # Convert the plate_region to a PyTorch tensor and move it to the same device as the model
            plate_region_tensor = torch.from_numpy(plate_region).to(device)
            ocr_thread = OCRThread(plate_region_tensor) # Pass the tensor to the OCRThread
            ocr_thread.result_signal.connect(self.handle_ocr_result)
            ocr_thread.start()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        convertToQtFormat = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        self.label_preview.setPixmap(QPixmap.fromImage(p))

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        # Close any other resources here
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
