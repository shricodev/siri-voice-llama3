import os
from datetime import datetime
from typing import NoReturn, Union

import cv2

import utils


def get_available_webcam() -> cv2.VideoCapture | None:
    # Assuming that we are checking the first 10 webcams.
    for index in range(10):
        web_cam = cv2.VideoCapture(index)
        if web_cam.isOpened():
            return web_cam

    return utils.exit_program(status_code=1, message="No webcams found.")


def capture_webcam_image() -> Union[str, NoReturn]:
    webcam = get_available_webcam()
    if webcam is None or not webcam.isOpened():
        return utils.exit_program(
            status_code=1, message="There was an error capturing the image."
        )

    webcam_folder_path = utils.get_path_to_folder(folder_type="webcam")

    if not os.path.exists(webcam_folder_path):
        os.makedirs(webcam_folder_path)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    image_filename = f"webcam_{timestamp}.png"

    _, frame = webcam.read()

    image_file_path = os.path.join(webcam_folder_path, image_filename)

    cv2.imwrite(image_file_path, frame)

    webcam.release()

    return image_file_path
