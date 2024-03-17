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

        self.placeholder_box = None
        self.white_label = None
        self.cam_label = None
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.model = load_model('ASL_model.h5')
        self.speak_engine = pyttsx3.init(driverName="sapi5")
        self.speak_engine.setProperty('rate', 100)
        voices = self.speak_engine.getProperty("voices")
        for voice in voices:
            self.speak_engine.setProperty('voice', voice.id)
        self.init_ui()
        self.char = ""  # This stores prediction

        # Start updating the image with the webcam stream
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(100)  # Update every 100 milliseconds (adjust as needed)

        self.white_image_path = "static_images/white.jpg"
        self.white_image = cv2.imread(self.white_image_path)
        self.model_input = self.white_image

        self.mp_drawing = mp_drawing
        self.mp_hands = mp_hands
        self.hands = mp_hands.Hands(max_num_hands=1)
        self.pts = None

        self.update_image()

        self.speak()
        self.predict(self.model_input)
        self.__del__()

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
        right_image = QPixmap(r"static_images/signs.png")
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
        self.placeholder_box.setStyleSheet(
            "font-size: 60px; border: 1px solid black; color: red")  # Set text color to red
        fourth_row_layout.addWidget(self.placeholder_box)

        # Add a stretch item to push the speaker button to the right
        fourth_row_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Expanding))

        speak_button = QPushButton("Speak")
        speak_button.setIcon(QIcon("static_images/speaker.png"))  # Add your speaker icon path here
        speak_button.setFixedSize(100, 50)  # Adjust size of the button
        speak_button.clicked.connect(self.speak)
        fourth_row_layout.addWidget(speak_button)

        main_layout.addLayout(fourth_row_layout, 3)  # Set the stretch factor to 3 for bigger height

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
            frame_qimage = QImage(frame_rgb.data, frame_width, frame_height, frame_width * frame_channels,
                                  QImage.Format.Format_RGB888)

            # Update the webcam stream image
            frame_pixmap = QPixmap.fromImage(frame_qimage)
            self.cam_label.setPixmap(frame_pixmap)

            # Process the frame for hand landmarks and prediction
            results = self.hands.process(frame_rgb)

            white = cv2.imread("static_images/white.jpg")

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(white, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3))
                    self.pts = hand_landmarks.landmark
                    print("Self.pts is:", self.pts)

            cv2.imshow("frame", frame)
            cv2.imshow("1", white)  # Display the hand skeleton image

            # cv2.imshow("frame", frame)
            # cv2.imshow("1", white)
            skeleton_pixmap = QPixmap.fromImage(
                QImage(white.data, white.shape[1], white.shape[0], QImage.Format.Format_RGB888))
            self.white_label.setPixmap(skeleton_pixmap)
            self.model_input = white

            self.predict(self.model_input)
            self.placeholder_box.setText(self.char)

            interrupt = cv2.waitKey(1)
            if interrupt & 0xFF == 27:
                exit(1)

        except Exception:
            print("==", traceback.format_exc())

    def speak(self):
        self.speak_engine.say(self.char)
        self.speak_engine.runAndWait()

    @staticmethod
    def distance(x, y):
        return math.sqrt(((x.x - y.x) ** 2) + ((x.y - y.y) ** 2))

    def predict(self, image):
        white = image.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]
        print(pl)

        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6].x < self.pts[8].x and self.pts[10].x < self.pts[12].x and self.pts[14].x <
                    self.pts[16].x and self.pts[18].x < self.pts[20].x):
                ch1 = 0
                print("00000")

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if self.pts[5].x < self.pts[4].x:
                ch1 = 0
                print("++++++++++++++++++")
                print("00000")

        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[0].x > self.pts[8].x and self.pts[0].x > self.pts[4].x and self.pts[0].x > self.pts[12].x
                    and self.pts[0].x > self.pts[16].x and self.pts[0].x > self.pts[20].x)
                    and self.pts[5].x > self.pts[4].x):
                ch1 = 2
                print("22222")

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2
                print("22222")

        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6].y > self.pts[8].y and self.pts[14].y < self.pts[16].y and self.pts[18].y < self.pts[20].y
                    and self.pts[0].x < self.pts[8].x and self.pts[0].x < self.pts[12].x and
                    self.pts[0].x < self.pts[16].x and self.pts[0].x < self.pts[20].x):
                ch1 = 3
                print("33333c")

        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4].x > self.pts[0].x:
                ch1 = 3
                print("33333b")

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2].y + 15 < self.pts[16].y:
                ch1 = 3
                print("33333a")

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4
                # print("44444")

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6].y > self.pts[8].y and self.pts[10].y < self.pts[12].y and self.pts[14].y < self.pts[
                16].y and self.pts[18].y <
                    self.pts[20].y):
                ch1 = 4
                # print("44444")

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4].x < self.pts[0].x):
                ch1 = 4
                # print("44444")

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1].x < self.pts[12].x):
                ch1 = 4
                # print("44444")

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6].y > self.pts[8].y and self.pts[10].y < self.pts[12].y and self.pts[14].y < self.pts[16].y
                    and self.pts[18].y < self.pts[20].y) and self.pts[4].y > self.pts[10].y:
                ch1 = 5
                print("55555b")

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4].y + 17 > self.pts[8].y and self.pts[4].y + 17 > self.pts[12].y and self.pts[4].y + 17 > self.pts[16].y
                    and self.pts[4].y + 17 > self.pts[20].y):
                ch1 = 5
                print("55555a")

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4].x > self.pts[0].x:
                ch1 = 5
                # print("55555")

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0].x < self.pts[8].x and self.pts[0].x < self.pts[12].x and self.pts[0].x < self.pts[16].x
                    and self.pts[0].x < self.pts[20].x):
                ch1 = 5
                # print("55555")

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3].x < self.pts[0].x:
                ch1 = 7
                # print("77777")

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6].y < self.pts[8].y:
                ch1 = 7
                # print("77777")

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18].y > self.pts[20].y:
                ch1 = 7
                # print("77777")

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5].x > self.pts[16].x:
                ch1 = 6
                print("666661")

        # condition for [yj][x]
        print("2222  ch1=+++++++++++++++++", ch1, ",", ch2)
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18].y < self.pts[20].y and self.pts[8].y < self.pts[10].y:
                ch1 = 6
                print("666662")

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6
                print("666663")

        # con for [l][x]
        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6
                print("666664")

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5].x - self.pts[4].x - 15 > 0:
                ch1 = 6
                print("666665")

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6].y > self.pts[8].y and self.pts[10].y > self.pts[12].y and self.pts[14].y > self.pts[16].y
                    and self.pts[18].y > self.pts[20].y):
                ch1 = 1
                print("111111")

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6].y < self.pts[8].y and self.pts[10].y > self.pts[12].y and self.pts[14].y > self.pts[
                16].y and
                    self.pts[18].y > self.pts[20].y):
                ch1 = 1
                print("111112")

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10].y > self.pts[12].y and self.pts[14].y > self.pts[16].y and
                    self.pts[18].y > self.pts[20].y):
                ch1 = 1
                print("111112")

        # con for [d][pqz]
        fg = 19
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6].y > self.pts[8].y and self.pts[10].y < self.pts[12].y and self.pts[14].y < self.pts[16].y and
                    self.pts[18].y < self.pts[20].y) and (self.pts[2].x < self.pts[0].x) and self.pts[4].y > self.pts[14].y):
                ch1 = 1
                print("111113")

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6].y > self.pts[8].y and self.pts[10].y < self.pts[12].y
                    and self.pts[14].y < self.pts[16].y and self.pts[18].y < self.pts[20].y):
                ch1 = 1
                print("1111993")

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6].y > self.pts[8].y and self.pts[10].y < self.pts[12].y and self.pts[14].y < self.pts[
                16].y and
                 self.pts[18].y < self.pts[20].y) and (self.pts[2].x < self.pts[0].x) and self.pts[14].y < self.pts[4].y):
                ch1 = 1
                print("1111mmm3")

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5].x - self.pts[4].x - 15 < 0:
                ch1 = 1
                print("1111140")

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6].y < self.pts[8].y and self.pts[10].y < self.pts[12].y and self.pts[14].y < self.pts[16].y and
                     self.pts[18].y > self.pts[20].y)):
                ch1 = 1
                print("111114")

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4].x < self.pts[5].x + 15) and (
                    (self.pts[6].y < self.pts[8].y and self.pts[10].y < self.pts[12].y and self.pts[14].y < self.pts[
                        16].y and
                     self.pts[18].y > self.pts[20].y)):
                ch1 = 7
                print("111114lll;;p")

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6].y > self.pts[8].y and self.pts[10].y > self.pts[12].y and self.pts[14].y < self.pts[16].y
                 and self.pts[18].y < self.pts[20].y)) and self.pts[4].y > self.pts[14].y:
                ch1 = 1
                print("111115")

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if (not (self.pts[0].x + fg < self.pts[8].x and self.pts[0].x + fg < self.pts[12].x and self.pts[0].x +
                    fg < self.pts[16].x and self.pts[0].x + fg < self.pts[20].x) and
                    not (self.pts[0].x > self.pts[8].x and self.pts[0].x > self.pts[12].x
                    and self.pts[0].x > self.pts[16].x and self.pts[0].x > self.pts[20].x)
                    and self.distance(self.pts[4], self.pts[11]) < 50):
                ch1 = 1
                print("111116")

        # con for [w]
        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6].y > self.pts[8].y and self.pts[10].y > self.pts[12].y and self.pts[14].y > self.pts[16].y:
                ch1 = 1
                print("1117")

        if ch1 == 0:
            ch1 = 'S'
            if (self.pts[4].x < self.pts[6].x and self.pts[4].x < self.pts[10].x and self.pts[4].x < self.pts[14].x
                    and self.pts[4].x < self.pts[18].x):
                ch1 = 'A'
            if (self.pts[6].x < self.pts[4].x < self.pts[10].x and self.pts[4].x < self.pts[14].x
                    and self.pts[4].x < self.pts[18].x and self.pts[4].y < self.pts[14].y and self.pts[4].y < self.pts[18].y):
                ch1 = 'T'
            if (self.pts[4].y > self.pts[8].y and self.pts[4].y > self.pts[12].y and self.pts[4].y > self.pts[16].y
                    and self.pts[4].y > self.pts[20].y):
                ch1 = 'E'
            if (self.pts[4].x > self.pts[6].x and self.pts[4].x > self.pts[10].x and self.pts[4].x > self.pts[14].x
                    and self.pts[4].y < self.pts[18].y):
                ch1 = 'M'
            if (self.pts[4].x > self.pts[6].x and self.pts[4].x > self.pts[10].x and self.pts[4].y < self.pts[18].y
                    and self.pts[4].y < self.pts[14].y):
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4].x > self.pts[12].x and self.pts[4].x > self.pts[16].x and self.pts[4].x > self.pts[20].x:
                if self.pts[8].y < self.pts[5].y:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6].y > self.pts[8].y and self.pts[10].y > self.pts[12].y and self.pts[14].y > self.pts[16].y
                    and self.pts[18].y > self.pts[20].y):
                ch1 = 'B'
            if (self.pts[6].y > self.pts[8].y and self.pts[10].y < self.pts[12].y and self.pts[14].y < self.pts[16].y
                    and self.pts[18].y < self.pts[20].y):
                ch1 = 'D'
            if (self.pts[6].y < self.pts[8].y and self.pts[10].y > self.pts[12].y and self.pts[14].y > self.pts[16].y
                    and self.pts[18].y > self.pts[20].y):
                ch1 = 'F'
            if (self.pts[6].y < self.pts[8].y and self.pts[10].y < self.pts[12].y and self.pts[14].y < self.pts[16].y
                    and self.pts[18].y > self.pts[20].y):
                ch1 = 'I'
            if (self.pts[6].y > self.pts[8].y and self.pts[10].y > self.pts[12].y and self.pts[14].y > self.pts[16].y
                    and self.pts[18].y < self.pts[20].y):
                ch1 = 'W'
            if (self.pts[6].y > self.pts[8].y and self.pts[10].y > self.pts[12].y and self.pts[14].y < self.pts[16].y
                and self.pts[18].y < self.pts[20].y) and self.pts[4].y < self.pts[9].y:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6].y > self.pts[8].y and self.pts[10].y > self.pts[12].y and self.pts[14].y < self.pts[16].y
                    and self.pts[18].y < self.pts[20].y):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6].y > self.pts[8].y and self.pts[10].y > self.pts[12].y and self.pts[14].y < self.pts[16].y
                    and self.pts[18].y < self.pts[20].y) and (self.pts[4].y > self.pts[9].y):
                ch1 = 'V'

            if (self.pts[8].x > self.pts[12].x) and (
                    self.pts[6].y > self.pts[8].y and self.pts[10].y > self.pts[12].y and self.pts[14].y < self.pts[16].y
                    and self.pts[18].y < self.pts[20].y):
                ch1 = 'R'

        if ch1 == 1 or ch1 == 'E' or ch1 == 'S' or ch1 == 'X' or ch1 == 'Y' or ch1 == 'B':
            if (self.pts[6].y > self.pts[8].y and self.pts[10].y < self.pts[12].y and self.pts[14].y < self.pts[16].y
                    and self.pts[18].y > self.pts[20].y):
                ch1 = " "

        self.char = ch1

    def __del__(self):
        # Clean up resources or perform final tasks here
        print("Closing Application...")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GUIApp()
    window.show()
    sys.exit(app.exec())
