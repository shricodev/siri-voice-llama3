import os
import re
import time
from datetime import datetime
from typing import List

import google.generativeai as genai
import pyaudio
import speech_recognition as sr
from faster_whisper import WhisperModel
from groq import Groq
from groq.types.chat import ChatCompletionMessageParam
from openai import OpenAI
from PIL import Image, ImageGrab

import clipboard
import utils
import webcam


class Siri:
    def __init__(
        self, groq_api_key: str, google_gen_ai_api_key: str, openai_api_key: str
    ) -> None:
        self.groq_client = Groq(api_key=groq_api_key)
        self.openai_client = OpenAI(api_key=openai_api_key)

        genai_generation_config = genai.GenerationConfig(
            temperature=0.7, top_p=1, top_k=1, max_output_tokens=2048
        )
        genai.configure(api_key=google_gen_ai_api_key)
        self.genai_model = genai.GenerativeModel(
            "gemini-1.5-flash-latest",
            generation_config=genai_generation_config,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ],
        )

        self.conversation: List[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": (
                    "You are a multi-modal AI voice assistant. Your user may have attached a photo (screenshot or webcam capture) "
                    "for context, which has already been processed into a detailed text prompt. This will be attached to their transcribed "
                    "voice input. Generate the most relevant and factual response by carefully considering all previously generated text "
                    "before adding new information. Do not expect or request additional images; use the provided context if available. "
                    "Ensure your responses are clear, concise, and relevant to the ongoing conversation, avoiding any unnecessary verbosity."
                ),
            }
        ]

        total_cpu_cores = os.cpu_count() or 1

        self.audio_transcription_model = WhisperModel(
            device="cpu",
            compute_type="int8",
            model_size_or_path="base",
            cpu_threads=total_cpu_cores // 2,
            num_workers=total_cpu_cores // 2,
        )

        self.speech_recognizer = sr.Recognizer()
        self.mic_audio_source = sr.Microphone()
        self.wake_word = "siri"

    def transcribe_audio_to_text(self, audio_path: str) -> str:
        segments, _ = self.audio_transcription_model.transcribe(audio_path)
        return "".join(segment.text for segment in segments)

    def extract_prompt(self, transcribed_text: str) -> str | None:
        pattern = rf"\b{re.escape(self.wake_word)}[\s,.?!]*([A-Za-z0-9].*)"
        regex_match = re.search(
            pattern=pattern, string=transcribed_text, flags=re.IGNORECASE
        )

        if regex_match is None:
            return None

        return regex_match.group(1).strip()

    def start_listening(self) -> None:
        with self.mic_audio_source as mic:
            self.speech_recognizer.adjust_for_ambient_noise(source=mic, duration=2)

        print("LISTENING...\n")
        print("SAY THE WAKE WORD WITH YOUR PROMPT\n")

        self.speech_recognizer.listen_in_background(
            source=self.mic_audio_source, callback=self.handle_audio_processing
        )

        while True:
            time.sleep(0.5)

    def generate_chat_response_with_groq(
        self, prompt: str, image_context: str | None
    ) -> str:
        if image_context:
            prompt = f"USER_PROMPT: {prompt}\n\nIMAGE_CONTEXT: {image_context}"

        self.conversation.append({"role": "user", "content": prompt})

        completion = self.groq_client.chat.completions.create(
            messages=self.conversation, model="llama-3.1-8b-instant"
        )

        ai_response = completion.choices[0].message.content

        self.conversation.append({"role": "assistant", "content": ai_response})

        return ai_response or "Sorry, I'm not sure how to respond to that."

    def text_to_speech(self, text: str) -> None:
        stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16, channels=1, rate=24000, output=True
        )
        stream_start = False

        with self.openai_client.audio.speech.with_streaming_response.create(
            model="tts-1", voice="nova", response_format="pcm", input=text
        ) as openai_response:
            silence_threshold = 0.1
            for chunk in openai_response.iter_bytes(chunk_size=1024):
                if stream_start:
                    stream.write(chunk)

                elif max(chunk) > silence_threshold:
                    stream.write(chunk)
                    stream_start = True

    def select_assistant_action(self, prompt: str) -> str:
        system_prompt_message = (
            "You are an AI model tasked with selecting the most appropriate action for a voice assistant. Based on the user's prompt, "
            "choose one of the following actions: ['extract clipboard', 'take screenshot', 'delete screenshot', 'capture webcam', 'None']. "
            "Assume the webcam is a standard laptop webcam facing the user. Provide only the action without explanations or additional text. "
            "Respond strictly with the most suitable option from the list."
        )
        function_conversation: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt_message},
            {"role": "user", "content": prompt},
        ]

        completion = self.groq_client.chat.completions.create(
            messages=function_conversation, model="llama-3.1-8b-instant"
        )

        ai_response = completion.choices[0].message.content
        print(f"TASK: {ai_response}\n")

        return ai_response or "None"

    def capture_screenshot(self) -> str:
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

    def analyze_image_prompt(self, prompt: str, image_path: str) -> str:
        image = Image.open(image_path)
        prompt = (
            "You are an image analysis AI tasked with extracting semantic meaning from images to assist another AI in "
            "generating a user response. Your role is to analyze the image based on the user's prompt and provide all relevant, "
            "objective data without directly responding to the user. Focus solely on interpreting the image in the context of "
            f"the user’s request and relay that information for further processing. \nUSER_PROMPT: {prompt}"
        )
        genai_response = self.genai_model.generate_content([prompt, image])
        return genai_response.text

    def handle_audio_processing(self, recognizer, audio):
        audio_prompt_file_path = "prompt.wav"
        with open(audio_prompt_file_path, "wb") as f:
            f.write(audio.get_wav_data())

        transcribed_text = self.transcribe_audio_to_text(
            audio_path=audio_prompt_file_path
        )
        parsed_prompt = self.extract_prompt(transcribed_text=transcribed_text)

        if parsed_prompt:
            print(f"USER: {parsed_prompt}\n")
            selected_assistant_action = self.select_assistant_action(
                prompt=parsed_prompt
            )

            if "take screenshot" in selected_assistant_action:
                image_path = self.capture_screenshot()
                image_analysis_result = self.analyze_image_prompt(
                    prompt=parsed_prompt, image_path=image_path
                )

            elif "delete screenshot" in selected_assistant_action:
                utils.remove_last_screenshot()
                image_analysis_result = None

            elif "capture webcam" in selected_assistant_action:
                image_path = webcam.capture_webcam_image()
                image_analysis_result = self.analyze_image_prompt(
                    prompt=parsed_prompt, image_path=image_path
                )

            elif "extract clipboard" in selected_assistant_action:
                clipboard_content = clipboard.get_clipboard_text()
                parsed_prompt = (
                    f"{parsed_prompt}\n\nCLIPBOARD_CONTENT: {clipboard_content}"
                )
                image_analysis_result = None

            else:
                image_analysis_result = None

            response = self.generate_chat_response_with_groq(
                prompt=parsed_prompt, image_context=image_analysis_result
            )
            print(f"ASSISTANT: {response}\n")
            self.text_to_speech(text=response)
