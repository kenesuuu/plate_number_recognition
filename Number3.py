import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QWidget, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import numpy as np
import easyocr
import os  # Import os module to handle file operations

# Load the pre-trained Haar cascade classifier for license plate detection
n_plate_detector = cv2.CascadeClassifier("C:/Users/Caffiene/Documents/11PlateNumber/plate_number_recognition/model/haarcascade_russian_plate_number.xml")

class CameraThread(QThread):
    frame_available = pyqtSignal(np.ndarray)
    frame_skip = 5 # Process every 5th frame

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
        self.record_button.clicked.connect(self.record_details)
        widget.setLayout(layout)
        self.camera_thread = None
        self.reader = easyocr.Reader(['en'], gpu=True) # Initialize OCR reader with GPU support
        self.recorded_details = []
        self.save_folder = "recorded_details"  # Folder name to save recorded details

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
            plate_text = self.read_license_plate(plate_region) # Read license plate text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Store detected plate details
            self.recorded_details.append((plate_text, plate_region.copy()))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        convertToQtFormat = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        self.label_preview.setPixmap(QPixmap.fromImage(p))

    def read_license_plate(self, plate_region):
        gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_plate, 150, 255, cv2.THRESH_BINARY_INV)
        plate_text = self.reader.readtext(thresh, detail=0, paragraph=False)
        if plate_text:
            return plate_text[0][-2] # Assuming the first recognized text is the license plate number
        else:
            return "Not Recognized"

    def record_details(self):
        if self.recorded_details:
            for i, (plate_text, plate_region) in enumerate(self.recorded_details):
                filename = os.path.join(self.save_folder, f"detected_plate_{i}.jpg")
                cv2.imwrite(filename, plate_region)
                print(f"Plate Number: {plate_text} - Details saved as {filename}")

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        # Close any other resources here

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
