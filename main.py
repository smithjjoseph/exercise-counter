#!/usr/bin/python3
"""!
@file	main.py
@brief	Program entry point for exercise counter
@author	Joseph Smith
"""

import sys
import cv2
import numpy as np
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_draw_utils
import mediapipe.python.solutions.drawing_styles as mp_draw_styles
from pathlib import Path

WINDOW_NAME = 'Pose Detection'
WEBCAM_RES = (640, 480)
SCALING = 1

pose = mp_pose.Pose(smooth_landmarks=True,
                    min_detection_confidence=0.8,
                    min_tracking_confidence=0.5)


class PoseStream:
    def draw_pose_landmarks(self, image, pose_landmarks) -> None:
        if not pose_landmarks:
            return

        # for landmark in pose_landmarks.landmark:
        mp_draw_utils.draw_landmarks(
            image,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw_styles.get_default_pose_landmarks_style())


    def main(self) -> None:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, image = cap.read()

            # Avoids empty images from causing errors
            if not success or image is None:
                continue

            image = cv2.flip(image, 1) # Flip in y-axis to mirror
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            pose_landmarks = results.pose_landmarks
            self.draw_pose_landmarks(image, pose_landmarks)

            cv2.namedWindow(WINDOW_NAME ,cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, int(WEBCAM_RES[0]/SCALING),
                             int(WEBCAM_RES[1]/SCALING))
            cv2.imshow(WINDOW_NAME, image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    pose_stream = PoseStream()
    pose_stream.main()