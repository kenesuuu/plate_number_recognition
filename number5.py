import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QWidget, QPushButton, QFileDialog, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QMovie, QPalette, QColor, QIcon
from PyQt5.QtCore import Qt, QTimer
import easyocr
import os
import torch

# Constants for clarity and easy adjustment
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
SCALE_FACTOR = 1.05
MIN_NEIGHBORS = 7
CANNY_THRESHOLD1 = 10
CANNY_THRESHOLD2 = 200
GAUSSIAN_BLUR_SIZE = (5, 5)
GAUSSIAN_BLUR_SIGMA = 0
CONFIDENCE_THRESHOLD = 0.5

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("License Plate Detector")
        self.setWindowIcon(QIcon('carlogo3.png'))
        
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        
        self.set_dark_theme()
        
        self.background_label = QLabel(self)
        self.background_label.setGeometry(0, 0, self.width(), self.height())
        self.background_label.setAutoFillBackground(True)
        self.background_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.background_movie = QMovie("car.gif")
        self.background_label.setMovie(self.background_movie)
        self.background_movie.start()
        
        layout.addWidget(self.background_label)
        
        self.label_preview = QLabel(self)
        self.label_preview.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        layout.addWidget(self.label_preview)
        
        self.start_button = QPushButton("Start Camera")
        layout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start_camera)
        
        self.stop_button = QPushButton("Stop Camera")
        layout.addWidget(self.stop_button)
        self.stop_button.setVisible(False)
        
        self.gallery_button = QPushButton("Open Gallery")
        layout.addWidget(self.gallery_button)
        self.gallery_button.clicked.connect(self.open_gallery)
        
        widget.setLayout(layout)
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.detected_plates = set()
        self.recorded_plates_dir = "recorded_details"

    def set_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        self.setPalette(palette)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #333;
            }
            QPushButton {
                background-color: #444;
                color: white;
                border: none;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QLabel {
                color: white;
            }
        """)
    
    def start_camera(self):
        self.background_label.hide()
        video_source = 0
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 / 30)
        
        self.start_button.setVisible(False)
        self.stop_button.setVisible(True)
        try:
            self.start_button.clicked.disconnect()
        except Exception:
            pass
        self.stop_button.clicked.connect(self.stop_camera)
        
    def stop_camera(self):
        self.background_label.show()
        self.label_preview.hide()
        if self.cap:
            self.cap.release()
        QTimer.singleShot(0, self.adjustSize)
        
        self.stop_button.setVisible(False)
        self.start_button.setVisible(True)
        
        try:
            self.stop_button.clicked.disconnect()
        except Exception:
            pass
        
        self.start_button.clicked.connect(self.start_camera)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plates = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml").detectMultiScale(gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS)
            for (x, y, w, h) in plates:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                license_plate = gray[y:y + h, x:x + w]
                blur = cv2.GaussianBlur(license_plate, GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIGMA)
                edged = cv2.Canny(blur, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
                result = self.reader.readtext(edged, detail=1, paragraph=False)
                if result and len(result) > 0:
                    try:
                        text = result[0][1]
                        prob = result[0][2]
                        if prob > CONFIDENCE_THRESHOLD and text not in self.detected_plates:
                            self.detected_plates.add(text)
                            print(f"Detected and recorded: {text} (Confidence: {prob})")
                            license_plate_image = frame[y:y + h, x:x + w]
                            cv2.imwrite(os.path.join(self.recorded_plates_dir, f"{text}.jpg"), license_plate_image)
                            with open(os.path.join(self.recorded_plates_dir, f"{text}.txt"), 'w') as file:
                                file.write(text)
                    except IndexError:
                        print("Error: Unexpected result structure.")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            convertToQtFormat = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
            p = convertToQtFormat.scaled(CAMERA_WIDTH, CAMERA_HEIGHT, Qt.KeepAspectRatio)
            self.label_preview.setPixmap(QPixmap.fromImage(p))

    def open_gallery(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "", "Images (*.png *.xpm *.jpg)", options=options)
        if files:
            for file in files:
                pixmap = QPixmap(file)
                # Display the image in your application, e.g., in a QLabel
                # You might want to create a separate QLabel or a QListWidget to display multiple images
                # For demonstration, let's just print the file path
                print(f"Opened image: {file}")

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
