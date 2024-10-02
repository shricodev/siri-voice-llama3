import os
import sys
from datetime import datetime
from typing import List

from groq import Groq
from groq.types.chat import ChatCompletionMessageParam
from PIL import ImageGrab

# Add the src directory to the module search path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src import setup, utils, webcam

api_keys = setup.get_credentials()

groq_api_key, google_gen_ai_key, openai_api_key = api_keys

groq_client = Groq(api_key=groq_api_key)


def groq_prompt(prompt: str) -> str:
    conversation: List[ChatCompletionMessageParam] = [
        {"role": "user", "content": prompt}
    ]

    completion = groq_client.chat.completions.create(
        messages=conversation, model="llama-3.1-8b-instant"
    )

    ai_response = completion.choices[0].message.content

    if ai_response is None:
        return "Sorry, I'm not sure how to respond to that."

    return ai_response


def function_call(prompt: str):
    sys_message = (
        "You are an AI model tasked with selecting the most appropriate action for a voice assistant. Based on the user's prompt, "
        "choose one of the following actions: ['extract clipboard', 'take screenshot', 'capture webcam', 'None']. "
        "Assume the webcam is a standard laptop webcam facing the user. Provide only the action without explanations or additional text. "
        "Respond strictly with the most suitable option from the list."
    )
    function_conversation: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": sys_message},
        {"role": "user", "content": prompt},
    ]

    completion = groq_client.chat.completions.create(
        messages=function_conversation, model="llama-3.1-8b-instant"
    )

    ai_response = completion.choices[0].message.content

    if ai_response is None:
        return "Sorry, I'm not sure how to respond to that."

    return ai_response


def take_screenshot() -> None:
    path_to_folder = utils.get_path_to_folder(folder_type="screenshot")

    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)

    screen = ImageGrab.grab()

    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    rgb_screenshot = screen.convert("RGB")

    file_name = f"screenshot_{current_time}.png"
    rgb_screenshot.save(os.path.join(path_to_folder, file_name), quality=20)


def delete_last_screenshot() -> None:
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


def webcam_capture():
    return webcam.webcam_capture()


def extract_clipboard():
    return None


# delete_last_screenshot()
take_screenshot()

# user_prompt = input("USER: ")
# function_response = function_call(prompt=user_prompt)
# print(f"FUNCTION: {function_response}")
# response = groq_prompt(prompt=user_prompt)
# print(f"AI: {response}")
