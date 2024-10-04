import os
import sys
from pathlib import Path
from typing import Literal, NoReturn

import utils


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
