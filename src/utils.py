import os
import sys
from pathlib import Path
from typing import Literal, NoReturn

import utils


def exit_program(status_code: int = 0, message: str = "") -> NoReturn:
    if message:
        print(f"ERROR: {message}\n")
    sys.exit(status_code)


def remove_last_screenshot() -> None:
    path_to_folder = utils.get_path_to_folder(folder_type="screenshot")

    if not os.path.exists(path_to_folder):
        return

    files = [
        file
        for file in os.listdir(path_to_folder)
        if os.path.isfile(os.path.join(path_to_folder, file)) and file.endswith(".png")
    ]
    if not files:
        return

    most_recent_file = max(
        files, key=lambda f: os.path.getctime(os.path.join(path_to_folder, f))
    )

    os.remove(os.path.join(path_to_folder, most_recent_file))


def get_path_to_folder(folder_type: Literal["webcam", "screenshot"]) -> str:
    base_path = f"{Path.home()}/Pictures/llama3.1/"

    if folder_type == "screenshot":
        return os.path.join(base_path, "Screenshots/")

    return os.path.join(base_path, "Webcam/")
