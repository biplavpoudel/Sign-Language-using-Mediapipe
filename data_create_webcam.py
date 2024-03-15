# import cv2
# import numpy as np
# import mediapipe as mp
# import os
# import traceback
#
# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands
#
# capture = cv2.VideoCapture(0)
# hands = mp_hands.Hands(max_num_hands=1)
#
# count = len(os.listdir(r"D:\Mediapipe ASL\MediaPipe ASL\Input\landmark_images\A\\"))
# c_dir = 'A'
#
# offset = 15
# step = 1
# flag = False
# suv = 0
#
# white = np.ones((400, 400), np.uint8) * 255
# cv2.imwrite("./white.jpg", white)
#
# while True:
#     try:
#         _, frame = capture.read()
#         frame = cv2.flip(frame, 1)
#
#         # Convert the image to RGB format for Mediapipe
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)
#
#         white = cv2.imread("./white.jpg")
#
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 pts = []
#                 for landmark in hand_landmarks.landmark:
#                     x = int(landmark.x * frame.shape[1])
#                     y = int(landmark.y * frame.shape[0])
#                     pts.append((x, y))
#
#                 os = ((400 - frame.shape[1]) // 2) - 15
#                 os1 = ((400 - frame.shape[0]) // 2) - 15
#                 for t in range(0, 4, 1):
#                     cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
#                              (0, 255, 0), 3)
#                 for t in range(5, 8, 1):
#                     cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
#                              (0, 255, 0), 3)
#                 for t in range(9, 12, 1):
#                     cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
#                              (0, 255, 0), 3)
#                 for t in range(13, 16, 1):
#                     cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
#                              (0, 255, 0), 3)
#                 for t in range(17, 20, 1):
#                     cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
#                              (0, 255, 0), 3)
#                 cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0), 3)
#                 cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0),
#                          3)
#                 cv2.line(white, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0),
#                          3)
#                 cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0), 3)
#                 cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0),
#                          3)
#
#                 skeleton0 = np.array(white)
#                 zz = np.array(white)
#                 for i in range(21):
#                     cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)
#
#                 skeleton1 = np.array(white)
#
#                 cv2.imshow("1", skeleton1)
#
#         frame = cv2.putText(frame, "dir=" + str(c_dir) + "  count=" + str(count), (50, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             1, (255, 0, 0), 1, cv2.LINE_AA)
#         cv2.imshow("frame", frame)
#         interrupt = cv2.waitKey(1)
#         if interrupt & 0xFF == 27:
#             # esc key
#             break
#
#         if interrupt & 0xFF == ord('n'):
#             c_dir = chr(ord(c_dir) + 1)
#             if ord(c_dir) == ord('Z') + 1:
#                 c_dir = 'A'
#             flag = False
#             count = len(
#                 os.listdir(r"D:\Mediapipe ASL\MediaPipe ASL\Input\landmark_images" + (c_dir) + "\\"))
#
#         if interrupt & 0xFF == ord('a'):
#             if flag:
#                 flag = False
#             else:
#                 suv = 0
#                 flag = True
#
#         print("=====", flag)
#         if flag:
#             if suv == 180:
#                 flag = False
#             if step % 3 == 0:
#                 cv2.imwrite(r"D:\Mediapipe ASL\MediaPipe ASL\Input\landmark_images" + (c_dir) + "\\" + str(count) + ".jpg",skeleton1)
#                 count += 1
#                 suv += 1
#             step += 1
#
#     except Exception:
#         print("==", traceback.format_exc())
#
# capture.release()
# cv2.destroyAllWindows()
#

import cv2
import numpy as np
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
import os
import traceback

mp_drawing = mp_drawing
mp_hands = mp_hands

capture = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=1)

count = len(os.listdir(r"D:\Mediapipe ASL\MediaPipe ASL\Input\landmark_images\A"))
c_dir = 'A'

offset = 15
step = 1
flag = False
suv = 0

white = np.ones((400, 400), np.uint8) * 255
cv2.imwrite("./white.jpg", white)

while True:
    try:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)

        # Convert the image to RGB format for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        white = cv2.imread("./white.jpg")

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(white, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3))

        frame = cv2.putText(frame, "dir=" + str(c_dir) + "  count=" + str(count), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        cv2.imshow("1", white)  # Display the hand skeleton image

        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            # esc key
            break

        if interrupt & 0xFF == ord('n'):
            c_dir = chr(ord(c_dir) + 1)
            if ord(c_dir) == ord('Z') + 1:
                c_dir = 'A'
            flag = False
            count = len(
                os.listdir(r"D:\Mediapipe ASL\MediaPipe ASL\Input\landmark_images\\" + (c_dir) + "\\"))

        if interrupt & 0xFF == ord('a'):
            if flag:
                flag = False
            else:
                suv = 0
                flag = True

        print("=====", flag)
        if flag:
            if suv == 180:
                flag = False
            if step % 3 == 0:
                cv2.imwrite(
                    r"D:\Mediapipe ASL\MediaPipe ASL\Input\landmark_images\\" + (c_dir) + "\\" + str(
                        count) + ".jpg",
                    white)  # Save the hand skeleton image
                count += 1
                suv += 1
            step += 1

    except Exception:
        print("==", traceback.format_exc())

capture.release()
cv2.destroyAllWindows()


