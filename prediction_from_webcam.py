import torch
import mediapipe as mp
import numpy as np
import math
import cv2
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
import time
import os, sys
import traceback
import pyttsx3
from keras.models import load_model
import enchant
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton
from PyQt6.QtWidgets import QMessageBox, QLabel, QMainWindow, QFrame
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt6.QtWidgets import QSizePolicy, QSpacerItem, QTextEdit
from PIL import Image, ImageTk
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
import cv2

class GUIApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.model = load_model('ASL_model.h5')
        self.speak_engine = pyttsx3.init(driverName="sapi5")
        self.speak_engine.setProperty('rate', 100)
        voices = self.speak_engine.getProperty("voices")
        for voice in voices:
            self.speak_engine.setProperty('voice', voice.id)
        self.init_ui()

        # Start updating the image with the webcam stream
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(100)  # Update every 100 milliseconds (adjust as needed)

        self.mp_drawing = mp_drawing
        self.mp_hands = mp_hands
        self.hands = mp_hands.Hands(max_num_hands=1)

        self.offset = 15
        self.step = 1
        self.flag = False
        self.suv = 0
        self.update_image()

        self.white_image_path = "./white.jpg"
        self.white_image = cv2.imread(self.white_image_path)

    def init_ui(self):
        self.setWindowTitle("Sign Language To Text Conversion")
        self.setGeometry(100, 100, 1920, 1080)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)  # Adjust spacing between rows

        # First row: Title label
        title_label = QLabel("American Sign Language Recognition System")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 36px; padding: 10px;")  # Adjust font size and padding
        main_layout.addWidget(title_label, 3)  # Set the stretch factor to 3 for bigger height

        # Second row: Image titles
        second_row_layout = QHBoxLayout()
        second_row_layout.setSpacing(20)  # Adjust spacing between columns

        # Left panel title
        webcam_title = QLabel("Webcam Stream")
        webcam_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        webcam_title.setStyleSheet("font-size: 18px; color: blue; padding: 10px;")  # Adjust padding
        second_row_layout.addWidget(webcam_title)

        # Middle panel title
        skeletons_title = QLabel("Mediapipe Skeletons")
        skeletons_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        skeletons_title.setStyleSheet("font-size: 18px; color: green; padding: 10px;")  # Adjust padding
        second_row_layout.addWidget(skeletons_title)

        # Right panel title
        asl_title = QLabel("ASL Signs")
        asl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        asl_title.setStyleSheet("font-size: 18px; color: red; padding: 10px;")  # Adjust padding
        second_row_layout.addWidget(asl_title)

        main_layout.addLayout(second_row_layout, 0)  # Set the stretch factor to 1 for smaller height

        # Third row: Image panels
        third_row_layout = QHBoxLayout()
        third_row_layout.setSpacing(10)  # Adjust spacing between columns

        # Left panel: Webcam Stream Image
        self.cam_label = QLabel()
        left_image = QPixmap("")
        left_image = left_image.scaled(640, 480, Qt.AspectRatioMode.IgnoreAspectRatio)
        self.cam_label.setPixmap(left_image)
        left_panel_layout = QHBoxLayout()
        left_panel_layout.addItem(QSpacerItem(40, 20))  # Add horizontal spacer for left padding
        left_panel_layout.addWidget(self.cam_label)
        third_row_layout.addLayout(left_panel_layout)

        # Middle panel: Mediapipe Skeletons Image
        self.white_label = QLabel()
        middle_image = QPixmap("")
        middle_image = middle_image.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)
        self.white_label.setPixmap(middle_image)
        third_row_layout.addWidget(self.white_label)

        # Right panel: ASL Signs Image
        right_panel = QLabel()
        right_image = QPixmap(r"D:\Mediapipe ASL\MediaPipe ASL\signs.png")
        right_image = right_image.scaled(500, 700, Qt.AspectRatioMode.KeepAspectRatio)
        right_panel.setPixmap(right_image)
        third_row_layout.addWidget(right_panel)

        main_layout.addLayout(third_row_layout, 5)  # Set the stretch factor to 5 for bigger height

        # Horizontal spacer between third and fourth rows
        horizontal_spacer = QSpacerItem(40, 40)  # Adjust the size as needed
        main_layout.addSpacerItem(horizontal_spacer)

        # Fourth row: Labels and button
        fourth_row_layout = QHBoxLayout()
        fourth_row_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)  # Align the layout to the left

        character_label = QLabel("Character:")
        character_label.setStyleSheet("font-size: 18px; padding-left: 20px;")
        fourth_row_layout.addWidget(character_label)

        # Placeholder box as QLabel
        self.placeholder_box = QLabel("")  # Replace "A" with your predicted character
        self.placeholder_box.setFixedSize(100, 100)  # Adjust the size of the box as needed
        self.placeholder_box.setAlignment(Qt.AlignmentFlag.AlignLeft)  # Align the placeholder box to the left
        self.placeholder_box.setStyleSheet("font-size: 60px; border: 1px solid black; color: red")  # Set text color to red
        fourth_row_layout.addWidget(self.placeholder_box)

        # Add a stretch item to push the speaker button to the right
        fourth_row_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Expanding))

        speak_button = QPushButton("Speak")
        speak_button.setIcon(QIcon("speaker.png"))  # Add your speaker icon path here
        speak_button.setFixedSize(100, 50)  # Adjust size of the button
        fourth_row_layout.addWidget(speak_button)

        main_layout.addLayout(fourth_row_layout, 3) # Set the stretch factor to 3 for bigger height

        main_layout.addItem(QSpacerItem(40, 40))  # Add vertical spacer below fourth row

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def update_image(self):
        try:
            _, frame = self.vs.read()
            frame = cv2.flip(frame, 1)

            # Convert the image to RGB format for Mediapipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_height, frame_width, frame_channels = frame_rgb.shape
            frame_qimage = QImage(frame_rgb.data, frame_width, frame_height, frame_width * frame_channels, QImage.Format.Format_RGB888)

            # Update the webcam stream image
            frame_pixmap = QPixmap.fromImage(frame_qimage)
            self.cam_label.setPixmap(frame_pixmap)

            # Process the frame for hand landmarks and prediction
            results = self.hands.process(frame_rgb)

            white = cv2.imread("./white.jpg")

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(white, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3))

            cv2.imshow("frame", frame)
            cv2.imshow("1", white)  # Display the hand skeleton image

            # cv2.imshow("frame", frame)
            # cv2.imshow("1", white)
            skeleton_pixmap = QPixmap.fromImage(QImage(white.data, white.shape[1], white.shape[0], QImage.Format.Format_RGB888))
            self.white_label.setPixmap(skeleton_pixmap)
            model_input = white.reshape(1, 400, 400, 3)
            prob = np.array(self.model.predict(model_input)[0], dtype='float32')
            ch1 = np.argmax(prob, axis=0)
            print(f"Predicted Output is: {str(ch1)}")

            self.placeholder_box.setText(str(ch1))

            interrupt = cv2.waitKey(1)
            if interrupt & 0xFF == 27:
                exit(1)

        except Exception:
            print("==", traceback.format_exc())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GUIApp()
    window.show()
    sys.exit(app.exec())

