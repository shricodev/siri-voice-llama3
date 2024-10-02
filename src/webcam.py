import os
from datetime import datetime

import cv2

import utils
from setup import exit_program


def find_working_webcam() -> cv2.VideoCapture | None:
    # Assuming that we are checking the first 10 webcams.
    for index in range(10):
        web_cam = cv2.VideoCapture(index)
        if web_cam.isOpened():
            return web_cam

    return exit_program(status_code=1)


def webcam_capture():
    web_cam = find_working_webcam()
    if web_cam is None or not web_cam.isOpened():
        print("ERROR: Unable to open the webcam")
        return exit_program(status_code=1)

    path_to_folder = utils.get_path_to_folder(folder_type="webcam")

    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)

    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"webcam_{current_time}.png"

    _, frame = web_cam.read()

    cv2.imwrite(os.path.join(path_to_folder, filename), frame)

    web_cam.release()


webcam_capture()
