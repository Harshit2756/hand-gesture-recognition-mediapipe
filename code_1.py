#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=int, default=0.5)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,  # Detect both hands simultaneously
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    mode = 0

    while True:
        fps = cvFpsCalc.get()

        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            combined_landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                combined_landmarks += calc_landmark_list(debug_image, hand_landmarks)

            pre_processed_landmark_list = pre_process_landmark(combined_landmarks)
            pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

            logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            if hand_sign_id == 2:  # Point gesture
                point_history.append(combined_landmarks[8])
            else:
                point_history.append([0, 0])

            finger_gesture_id = 0
            point_history_len = len(pre_processed_point_history_list)
            if point_history_len == (history_length * 2):
                finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

            finger_gesture_history.append(finger_gesture_id)
            most_common_fg_id = Counter(finger_gesture_history).most_common()

            debug_image = draw_info_text(debug_image, handedness, keypoint_classifier_labels[hand_sign_id],
                                         point_history_classifier_labels[most_common_fg_id[0][0]])
        else:
            point_history.append([0, 0])

        debug_image = draw_info(debug_image, fps, mode, number)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if 97 <= key <= 122:  # a ~ z
        number = key - 87  #
    if key == 92:  # \
        mode = 0
    if key == 47:  # /
        mode = 1
    if key == 46:  # .
        mode = 2
    return number, mode

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    # Normalize, etc.
    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    pre_processed_point_history_list = []
    for point in point_history:
        if sum(point) == 0:
            pre_processed_point_history_list.extend([0, 0])
        else:
            landmark_x = min(int(point[0] * image_width), image_width - 1)
            landmark_y = min(int(point[1] * image_height), image_height - 1)
            pre_processed_point_history_list.extend([landmark_x, landmark_y])
    return pre_processed_point_history_list

def logging_csv(number, mode, landmark_list, point_history_list):
    if number == -1:
        return
    mode_str = ""
    if mode == 0:
        mode_str = "point"
    elif mode == 1:
        mode_str = "history"
    elif mode == 2:
        mode_str = "both"

    with open(f'./data/{mode_str}/{mode_str}_{number}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(itertools.chain.from_iterable(landmark_list)) + point_history_list)

def draw_info_text(image, handedness, sign_label, gesture_label):
    # Draw text on the image
    cv.putText(image, f'Sign: {sign_label}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    cv.putText(image, f'Gesture: {gesture_label}', (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    return image

def draw_info(image, fps, mode, number):
    mode_str = ""
    if mode == 0:
        mode_str = "Point"
    elif mode == 1:
        mode_str = "History"
    elif mode == 2:
        mode_str = "Both"

    # Draw text on the image
    cv.putText(image, f'Mode: {mode_str}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    cv.putText(image, f'Number: {number}', (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    cv.putText(image, f'FPS: {fps:.2f}', (10, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    return image

if __name__ == "__main__":
    main()
