#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import copy
import csv
import itertools
import time
from collections import Counter
from collections import deque
import pygame
import cv2 as cv
import mediapipe as mp
import numpy as np
# from playsound import playsound
from model import KeyPointClassifier
from model import PointHistoryClassifier
from utils import CvFpsCalc


# / These functions are used to get the arguments from the command line which is used to set the camera device, width, height, and other parameters.
def get_args():
    parser = argparse.ArgumentParser()
    # / add_argument is used to add the arguments to the parser object.
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    # / here we are adding the argument to use the static image mode. 
    # / Static image mode is used to detect the hand in the static image (image without any movement).
    # / action parameter is used to set the action to be performed when the argument is passed. 
    # / here store_true is used to store the default value as True.
    parser.add_argument('--use_static_image_mode', action='store_true')

    # / min_detection_confidence tells if the confidence value is set to 0.7 then the hand will be detected only if the confidence value is greater than 0.7. 
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)

    # / min_tracking_confidence tells if the confidence value is set to 0.5 then the hand will be tracked only if the confidence value is greater than 0.5.
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()
    print(f'args: {args}')

    return args


def main():
    # * Argument parsing
    args = get_args()
    # Initialize pygame mixer
    pygame.mixer.init()

    # * Camera preparation
    # / VideoCapture object is used to capture the video from the camera. 
    # / The device parameter is used to set the camera device. 
    # / The width and height parameters are used to set the width and height of the captured video.
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # * Model load
    # / The Hand model is loaded using the mediapipe library.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,  # False
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,  # 0.7
        min_tracking_confidence=args.min_tracking_confidence,  # 0.5
    )

    # / KeyPointClassifier is a class that is used to classify the hand gestures. 
    keypoint_classifier = KeyPointClassifier()

    # / PointHistoryClassifier is a class that is used to classify the point history.
    point_history_classifier = PointHistoryClassifier()

    # * Read labels 
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as csvFile:
        keypoint_classifier_labels = csv.reader(csvFile)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # * FPS Measurement
    # / CvFpsCalc is a class that is used to calculate the frames per second.
    # / buffer_len te
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # * Coordinate history
    # ? / here we are using the deque class to store the history of the points, 16 means the length of the history i.e. the last 16 points are stored in the history.
    history_length = 16
    point_history = deque(maxlen=history_length)

    # * Finger gesture history
    # ? / 
    finger_gesture_history = deque(maxlen=history_length)

    # mode used to set the mode of the application. such as logging keypoint, logging point history, etc.
    mode = 0
    previous_hand_sign_id = 0
    previous_hand_sign_id_time = time.time()

    # / The while loop is used to capture the video from the camera and process the video frame by frame. 
    # / this is the main loop of the application.
    while True:
        fps = cvFpsCalc.get()

        # ~ Key input
        # /waitKey is used to wait for the key press event. 
        key = cv.waitKey(10)
        # ! Process Key (ESC: end) 
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # ~ Camera capture 
        # / The read() function is used to read the video frame from the camera. and return the frame (image) and bool(ret)
        # / image here is the frame captured from the camera. and ret is the boolean value that tells if the frame is captured or not. 
        ret, image = cap.read()
        if not ret:
            break
        # ~ Image processing
        # / The flip() function is used to flip the image horizontally() or vertically 
        # / here we are flipping the image horizontally by passing the value 1. to filp the image vertically we can pass 0. and to flip the image horizontally and vertically we can pass -1.
        image = cv.flip(image, 1)  # Mirror display
        # / deepcopy is used to create a deep copy of the image. 
        debug_image = copy.deepcopy(image)

        # ~ Detection implementation 
        # / cvtColor is used to convert the image from one color space to another color space.
        # / here we are converting the image from BGR color space to RGB color space as the mediapipe library uses the RGB color space and the OpenCV uses the BGR color space.
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # / image.flags.writeable is used to check if the image is writable i.e the image can be modified or not.
        image.flags.writeable = False
        # / The process() function is used to process the image and return the results such as the landmarks, handedness, etc.
        results = hands.process(image)
        # / image.flags.writeable is used to make the image writable i.e the image can be modified.
        image.flags.writeable = True

        # / here results.multi_hand_landmarks is used to check if the hand is detected or not.
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                print(landmark_list[0])

                # / Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                # ? Point history logging
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                # ? Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                # ? Point history logging
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                # / Drawing the bounding box, landmarks, and information on the image.
                # / draw the rectangle on the image
                # debug_image = draw_bounding_rect(True, debug_image, brect)
                # / Draw the hand line on the image
                # debug_image = draw_landmarks(debug_image, landmark_list)
                # / The draw_info_text[image(the image on which the information is to be drawn), brect(the bounding box coordinates), handedness(the handedness i.e left or right), hand_sign_text(the hand sign text), finger_gesture_text(the finger gesture text)] function is used to draw the information text on the image which has parameters
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],  # | / it has hand sign id
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )

                # / The play_audio[label(the label of the audio file to be played)] function is used to play the audio file which has parameters
                if hand_sign_id != previous_hand_sign_id:
                    previous_hand_sign_id = hand_sign_id
                    previous_hand_sign_id_time = time.time()
                else:
                    if time.time() - previous_hand_sign_id_time > 5.0:
                        draw_info_detected_text(debug_image, keypoint_classifier_labels[hand_sign_id])
                        play_audio(keypoint_classifier_labels[hand_sign_id])
            # draw_info_detected_text(debug_image, keypoint_classifier_labels[hand_sign_id])
        else:
            point_history.append([0, 0])

        try:
            if pygame.mixer.music.get_busy():
                draw_info_detected_text(debug_image, 'hello')
        except Exception as e:
            print(f"Error playing audio: {e}")

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection 
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def play_audio(label):
    try:
        # Assuming you have audio files saved with names corresponding to the labels
        audio_file = f"audio_files/{label}.mp3"

        # Load and play audio
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        # pygame.time.Clock()
        # draw_info_detected_text(debug_image, label)
        # pygame.time.wait(3000)    
        # while pygame.mixer.music.get_busy():
        #     pygame.event.poll()
        #     pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error playing audio: {e}")


# / This function is used to return the ascii value of the key pressed or the mode selected.
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if 97 <= key <= 122:  # a ~ z
        number = key - 87  #
    if key == 92:  # \
        mode = 0  # normal detection mode
    if key == 47:  # /
        mode = 1  # logging point mode
    # if key == 46:  # .
    #     mode = 2 # logging point history mode
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


# / This function is used to pre-process the landmarks i.e convert the landmarks to relative coordinates, convert the landmarks to a one-dimensional list, and normalize the landmarks.
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


# ? / This function is used to pre-process the point history.
def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list,
                ):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 35):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])

    # ? logging point history
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return



def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


# / This function is used to draw the information text on the image i.e the detected 
def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    # / The rectangle() function is used to draw the rectangle on the image.
    # / The parameters of the rectangle() function are image(the image on which the rectangle is to be drawn), pt1(the top-left corner of the rectangle), pt2(the bottom-right corner of the rectangle), color(the color of the rectangle), thickness(the thickness of the rectangle).
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture: h" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        # / The putText() function is used to draw the text on the image.
        # / The parameters of the putText() function are image(the image on which the text is to be drawn), text(the text to be drawn), org(the origin of the text), fontFace(the font of the text), fontScale(the font scale of the text), color(the color of the text), thickness(the thickness of the text), lineType(the type of the line of the text).
        cv.putText(image, "Finger Gesture: d" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)
    return image


def draw_info_detected_text(image, detected_text):
    if detected_text != "":
        cv.putText(image, "Detected Text:" + detected_text, (10, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4,
                   cv.LINE_AA)

    return image


# / This function is used to draw the point history on the image.
def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)
    return image


# / This function is used to draw the information on the image.
def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
