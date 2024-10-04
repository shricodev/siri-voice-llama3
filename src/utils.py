import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, NoReturn, Optional

from PIL import ImageGrab

import utils


def create_log_file_for_today(project_root_folder: str) -> str:
    today = datetime.today()

    # The year is always 4 digit and the month, day is always 2 digit using this format.
    year = today.strftime("%Y")
    month = today.strftime("%m")
    day = today.strftime("%d")

    base_folder = os.path.join(project_root_folder, "data", "chat_history", year, month)

    os.makedirs(base_folder, exist_ok=True)
    chat_log_file = os.path.join(base_folder, f"{day}.log")

    if not os.path.exists(chat_log_file):
        with open(chat_log_file, "w") as log_file:
            log_file.write("")

    return os.path.abspath(chat_log_file)


def log_chat_message(
    log_file_path: str,
    user_message: Optional[str] = None,
    ai_message: Optional[str] = None,
) -> None:
    # If neither of the message is given, return.
    if not user_message and not ai_message:
        return

    timestamp = datetime.now().strftime("[%H : %M]")

    with open(log_file_path, "a") as log_file:
        if user_message:
            log_file.write(f"{timestamp} - USER: {user_message}")

        if ai_message:
            log_file.write(f"{timestamp} - ASSISTANT: {ai_message}\n")

        log_file.write("\n")


def exit_program(status_code: int = 0, message: str = "") -> NoReturn:
    """
    Exit the program with an optional error message.

    Args:
        status_code (int): The exit status code. Defaults to 0 (success).
        message (str): An optional error message to display before exiting.
    """

    if message:
        print(f"ERROR: {message}\n")
    sys.exit(status_code)


def capture_screenshot() -> str:
    """
    Captures a screenshot and saves it to the designated folder.

    Returns:
        str: The file path of the saved screenshot.
    """

    screenshot_folder_path = utils.get_path_to_folder(folder_type="screenshot")

    if not os.path.exists(screenshot_folder_path):
        os.makedirs(screenshot_folder_path)

    screen = ImageGrab.grab()

    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    rgb_screenshot = screen.convert("RGB")

    image_filename = f"screenshot_{time_stamp}.png"
    image_file_path = os.path.join(screenshot_folder_path, image_filename)

    rgb_screenshot.save(image_file_path, quality=20)

    return image_file_path


def remove_last_screenshot() -> None:
    """
    Remove the most recent screenshot file from the designated screenshots folder.

    The function checks if the folder exists and if there are any .png files. If
    found, it deletes the most recently created screenshot.
    """

    folder_path = utils.get_path_to_folder(folder_type="screenshot")

    if not os.path.exists(folder_path):
        return

    files = [
        file
        for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file)) and file.endswith(".png")
    ]
    if not files:
        return

    most_recent_file = max(
        files, key=lambda f: os.path.getctime(os.path.join(folder_path, f))
    )

    os.remove(os.path.join(folder_path, most_recent_file))


def get_path_to_folder(folder_type: Literal["webcam", "screenshot"]) -> str:
    """
    Get the path to the specified folder type (webcam or screenshot).

    Args:
        folder_type (Literal["webcam", "screenshot"]): The type of folder to retrieve the path for.

    Returns:
        str: The path to the specified folder.
    """

    base_path = os.path.join(Path.home(), "Pictures", "llama3.1")

    if folder_type == "screenshot":
        return os.path.join(base_path, "Screenshots")

    elif folder_type == "webcam":
        return os.path.join(base_path, "Webcam")
