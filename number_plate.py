import cv2
import tkinter as tk
from PIL import Image, ImageTk

# Function to start the OpenCV script
def start_opencv():
    harcascade = "model/haarcascade_russian_plate_number.xml"
    cap = cv2.VideoCapture(0)

    cap.set(3, 640) # width
    cap.set(4, 480) # height

    min_area = 500
    count = 0

    while True:
        success, img = cap.read()

        plate_cascade = cv2.CascadeClassifier(harcascade)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        for (x, y, w, h) in plates:
            area = w * h

            if area > min_area:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                img_roi = img[y: y + h, x:x + w]
                cv2.imshow("ROI", img_roi)

        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)  # Create window with flag WINDOW_NORMAL
        cv2.imshow("Result", img)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("plates/scanned_img_" + str(count) + ".jpg", img_roi)
            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
            cv2.imshow("Result", img)
            cv2.waitKey(500)
            count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Create Tkinter window
root = tk.Tk()
root.title("OpenCV Application")

# Load and display an icon
try:
    icon = Image.open("icon.png")
    icon = icon.resize((100, 100))
    icon = ImageTk.PhotoImage(icon)
    label = tk.Label(root, image=icon)
    label.pack()
except FileNotFoundError:
    print("Icon file not found!")

# Button to start the OpenCV script
start_button = tk.Button(root, text="Start OpenCV", command=start_opencv)
start_button.pack()

root.mainloop()
