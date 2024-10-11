import os
from datetime import datetime
from pathlib import Path
from typing import NoReturn, Union

import cv2

import utils


def get_available_webcam() -> cv2.VideoCapture | None:
    """
    Checks for available webcams and returns the first one that is opened.

    This function attempts to open the first 10 webcam indices. If a webcam is found
    and successfully opened, it returns a VideoCapture object. If no webcams are found,
    it exits the program with an error message.

    Returns:
        cv2.VideoCapture: The opened webcam object.
        None: If no webcam is found, the program exits with an error message.
    """

    # Assuming that we are checking the first 10 webcams.
    for index in range(10):
        web_cam = cv2.VideoCapture(index)
        if web_cam.isOpened():
            return web_cam

    return utils.exit_program(status_code=1, message="No webcams found.")


def capture_webcam_image() -> Union[Path, NoReturn]:
    """
    Captures an image from the available webcam and saves it to the specified folder.

    This function first checks for an available webcam using `get_available_webcam`.
    If a webcam is successfully opened, it creates a folder for saving the images if
    it does not already exist, generates a timestamped filename, captures a frame,
    and saves the image to the specified folder. The function then releases the webcam.

    Returns:
        Path: The file path of the saved image.
        NoReturn: If there was an error capturing the image, the program exits with an error message.
    """

    webcam = get_available_webcam()
    if webcam is None or not webcam.isOpened():
        return utils.exit_program(
            status_code=1, message="There was an error capturing the image."
        )

    webcam_folder_path = utils.get_path_to_folder(folder_type="webcam")

    os.makedirs(webcam_folder_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    image_filename = f"webcam_{timestamp}.png"

    _, frame = webcam.read()

    image_file_path_str = os.path.join(webcam_folder_path, image_filename)

    cv2.imwrite(image_file_path_str, frame)

    webcam.release()

    return Path(image_file_path_str)
