import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QWidget, QPushButton, QFileDialog, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QMovie, QPalette, QColor, QIcon
from PyQt5.QtCore import Qt, QTimer
import easyocr
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained Haar cascade classifier for license plate detection
n_plate_detector = cv2.CascadeClassifier("C:/Users/Caffiene/Documents/11PlateNumber/plate_number_recognition/model/model.xml")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("License Plate Detector")
        
        # Set the window icon
        self.setWindowIcon(QIcon('C:/Users/Caffiene/Documents/11PlateNumber/plate_number_recognition/carlogo3.png'))
        
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        
        # Set dark theme for the application
        self.set_dark_theme()
        
        # Create a QLabel for the background
        self.background_label = QLabel(self)
        self.background_label.setGeometry(0, 0, self.width(), self.height())
        self.background_label.setAutoFillBackground(True)
        self.background_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Load the background movie
        self.background_movie = QMovie("C:\\Users\\Caffiene\\Documents\\11PlateNumber\\plate_number_recognition\\car.gif")
        self.background_label.setMovie(self.background_movie)
        self.background_movie.start()
        
        # Add the background label to the layout
        layout.addWidget(self.background_label)
        
        # Define label_preview before setting its size policy
        self.label_preview = QLabel(self)
        self.label_preview.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        layout.addWidget(self.label_preview)
        
        # Define start_button and stop_button before setting their visibility
        self.start_button = QPushButton("Start Camera")
        layout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start_camera)
        
        self.stop_button = QPushButton("Stop Camera")
        layout.addWidget(self.stop_button)
        self.stop_button.setVisible(False) # Initially, the stop button is not visible
        
        self.gallery_button = QPushButton("Open Gallery")
        layout.addWidget(self.gallery_button)
        self.gallery_button.clicked.connect(self.open_gallery)
        
        widget.setLayout(layout)
        self.reader = easyocr.Reader(['en'], gpu=True) # Initialize OCR reader with GPU support
        self.detected_plates = set() # Set to store unique license plate numbers
        self.recorded_plates_dir = "C:/Users/Caffiene/Documents/11PlateNumber/recorded_details"

    def set_dark_theme(self):
        # Set dark theme for the application
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
        app.setPalette(palette)
        
        # Set dark theme for the main window
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
        video_source = 0 # Use 0 for default camera
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 / 30) # 30 FPS
        
        # Hide the start button and show the stop button
        self.start_button.setVisible(False)
        self.stop_button.setVisible(True)
        try:
            self.start_button.clicked.disconnect()
        except Exception:
            pass
        self.stop_button.clicked.connect(self.stop_camera)
        

    def stop_camera(self):
        self.background_label.show()
        self.label_preview.hide() # Hide the camera view
        if self.cap:
            self.cap.release()
        # Force the window to resize to fit its contents
        QTimer.singleShot(0, self.adjustSize)
        
        # Hide the stop button and show the start button
        self.stop_button.setVisible(False)
        self.start_button.setVisible(True)
        
        # Disconnect the stop button from stop_camera and reconnect the start button to start_camera
        try:
            self.stop_button.clicked.disconnect()
        except Exception:
            pass
        
        # Reconnect the start button to start_camera
        self.start_button.clicked.connect(self.start_camera)



    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plates = n_plate_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)
            for (x, y, w, h) in plates:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                license_plate = gray[y:y + h, x:x + w]
                blur = cv2.GaussianBlur(license_plate, (5,5), 0)
                edged = cv2.Canny(blur, 10, 200)
                result = self.reader.readtext(edged, detail=1, paragraph=False)
                if result and len(result) > 0:
                    try:
                        text = result[0][1]
                        prob = result[0][2]
                        if prob > 0.5 and text not in self.detected_plates:
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
            p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            self.label_preview.setPixmap(QPixmap.fromImage(p))

    def open_gallery(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "", "Images (*.png *.xpm *.jpg)", options=options)
        if files:
            for file in files:
                pixmap = QPixmap(file)
                # Display the image in your application, e.g., in a QLabel
                # You might want to create```

                # Display the image in your application, e.g., in a QLabel
                # You might want to create a separate QLabel or a QListWidget to display multiple images

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
