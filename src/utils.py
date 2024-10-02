import os
from pathlib import Path
from typing import Literal


def get_path_to_folder(folder_type: Literal["webcam", "screenshot"]) -> str:
    base_path = f"{Path.home()}/Pictures/llama3.1/"

    if folder_type == "screenshot":
        return os.path.join(base_path, "Screenshots/")

    return os.path.join(base_path, "Webcam/")
