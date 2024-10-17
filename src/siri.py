import os
import re
import time
from pathlib import Path
from typing import List

import google.generativeai as genai
import pyttsx3
import speech_recognition as sr
from faster_whisper import WhisperModel
from groq import Groq
from groq.types.chat import ChatCompletionMessageParam
from gtts import gTTS
from openai import OpenAI
from PIL import Image
from pydub import AudioSegment
from pydub.playback import play

import utils
import webcam


class Siri:
    """
    A multi-modal AI voice assistant that responds to user prompts
    by processing voice commands and context from images or clipboard content.
    """

    def __init__(
        self,
        log_file_path: Path,
        project_root_folder_path: Path,
        groq_api_key: str,
        google_gen_ai_api_key: str,
        openai_api_key: str | None,
    ) -> None:
        """
        Initializes the Siri assistant with API clients for Groq, OpenAI, and Google Generative AI.

        Args:
            log_file_path (Path): Path to the log file.
            project_root_folder_path (Path): Root folder of the project.
            groq_api_key (str): API key for Groq.
            google_gen_ai_api_key (str): API key for Google Generative AI.
            openai_api_key (str): API key for OpenAI.
        """
        self.log_file_path = log_file_path
        self.project_root_folder_path = project_root_folder_path

        self.pyttsx3_engine = pyttsx3.init()

        self.groq_client = Groq(api_key=groq_api_key)
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Configure Google Generative AI model
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

        # Initialize conversation context for the AI
        self.conversation: List[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": (
                    "You are a multi-modal AI voice assistant. Your user may have attached a photo (screenshot or webcam capture) "
                    "for context, which has already been processed into a detailed text prompt. This will be attached to their transcribed "
                    "voice input. Generate the most relevant and factual response by carefully considering all previously generated text "
                    "before adding new information. Do not expect or request additional images; use the provided context if available. "
                    "Please do not include newlines in your response. Keep it all in one paragraph. "
                    "Ensure your responses are clear, concise, and relevant to the ongoing conversation, avoiding any unnecessary verbosity."
                ),
            }
        ]

        total_cpu_cores = os.cpu_count() or 1

        # Initialize the audio transcription model
        self.audio_transcription_model = WhisperModel(
            device="cpu",
            compute_type="int8",
            model_size_or_path="base",
            cpu_threads=total_cpu_cores // 2,
            num_workers=total_cpu_cores // 2,
        )

        # Initialize speech recognition components
        self.speech_recognizer = sr.Recognizer()
        self.mic_audio_source = sr.Microphone()
        self.wake_word = "siri"

    def transcribe_audio_to_text(self, audio_file_path: Path) -> str:
        """
        Transcribes audio from a file to text.

        Args:
            audio_file_path (Path): Path to the audio file.

        Returns:
            str: The transcribed text from the audio.
        """

        segments, _ = self.audio_transcription_model.transcribe(
            audio=str(audio_file_path)
        )
        return "".join(segment.text for segment in segments)

    def extract_prompt(self, transcribed_text: str) -> str | None:
        """
        Extracts the user's prompt from the transcribed text after the wake word.

        Args:
            transcribed_text (str): The transcribed text from audio input.

        Returns:
            str | None: The extracted prompt if found, otherwise None.
        """

        pattern = rf"\b{re.escape(self.wake_word)}[\s,.?!]*([A-Za-z0-9].*)"
        regex_match = re.search(
            pattern=pattern, string=transcribed_text, flags=re.IGNORECASE
        )

        if regex_match is None:
            return None

        return regex_match.group(1).strip()

    def listen(self) -> None:
        """
        Starts listening for the wake word and processes audio input in the background.
        """

        with self.mic_audio_source as mic:
            self.speech_recognizer.adjust_for_ambient_noise(source=mic, duration=2)

        self.speech_recognizer.listen_in_background(
            source=self.mic_audio_source, callback=self.handle_audio_processing
        )

        while True:
            time.sleep(0.5)

    def generate_chat_response_with_groq(
        self, prompt: str, image_context: str | None
    ) -> str:
        """
        Generates a response from the Groq model based on user input and optional image context.

        Args:
            prompt (str): The user's prompt.
            image_context (str | None): Optional image context for the response.

        Returns:
            str: The generated response from the assistant.
        """

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
        """
        Converts text to speech using OpenAI/pyttsx3/gTTS's text-to-speech API.

        Args:
            text (str): The text to convert to speech.
        """

        # USAGE: OpenAI approach (Use this if you have credits in your OpenAI account)

        # stream = pyaudio.PyAudio().open(
        #     format=pyaudio.paInt16, channels=1, rate=24000, output=True
        # )
        # stream_start = False
        #
        # with self.openai_client.audio.speech.with_streaming_response.create(
        #     model="tts-1", voice="nova", response_format="pcm", input=text
        # ) as openai_response:
        #     silence_threshold = 0.1
        #     for chunk in openai_response.iter_bytes(chunk_size=1024):
        #         if stream_start:
        #             stream.write(chunk)
        #
        #         elif max(chunk) > silence_threshold:
        #             stream.write(chunk)
        #             stream_start = True

        # USAGE: Pyttsx3 approach (Weak audio quality)

        # self.pyttsx3_engine.setProperty("volume", 1.0)
        # self.pyttsx3_engine.setProperty("rate", 125)
        #
        # voices = self.pyttsx3_engine.getProperty("voices")
        # self.pyttsx3_engine.setProperty("voice", voices[0].id)
        #
        # self.pyttsx3_engine.say(text)
        # self.pyttsx3_engine.runAndWait()
        #
        # self.pyttsx3_engine.stop()

        # gTTS approach (Stronger audio quality with Google TTS engine)
        # DOWNSIDE: Super slow. Also, there is a need to save it as `.mp3` and play it.
        tts = gTTS(text=text, lang="en", slow=False)

        response_folder_path = Path(
            os.path.abspath(
                os.path.join(self.project_root_folder_path, "data", "ai_response")
            )
        )

        os.makedirs(response_folder_path, exist_ok=True)

        response_audio_file_path = Path(
            os.path.join(response_folder_path, "ai_response_audio.mp3")
        )

        tts.save(response_audio_file_path)

        response_audio = AudioSegment.from_mp3(response_audio_file_path)
        play(response_audio)

        # After the audio is played, delete the audio file.
        if os.path.exists(response_audio_file_path):
            os.remove(response_audio_file_path)

    def select_assistant_action(self, prompt: str) -> str:
        """
        Determines the appropriate action for the assistant to take based on user input.

        Args:
            prompt (str): The user's prompt.

        Returns:
            str: The selected action for the assistant.
        """

        system_prompt_message = (
            "You are an AI model tasked with selecting the most appropriate action for a voice assistant. Based on the user's prompt, "
            "choose one of the following actions: ['extract clipboard', 'take screenshot', 'delete screenshot', 'capture webcam', 'generic']. "
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

        return ai_response or "generic"

    def analyze_image_prompt(self, prompt: str, image_path: Path) -> str:
        """
        Analyzes an image based on the user prompt to extract semantic information.

        Args:
            prompt (str): The user's prompt related to the image.
            image_path (Path): The path to the image file.

        Returns:
            str: The analysis result from the image based on the prompt.
        """

        image = Image.open(image_path)
        prompt = (
            "You are an image analysis AI tasked with extracting semantic meaning from images to assist another AI in "
            "generating a user response. Your role is to analyze the image based on the user's prompt and provide all relevant, "
            "objective data without directly responding to the user. Focus solely on interpreting the image in the context of "
            f"the user’s request and relay that information for further processing. \nUSER_PROMPT: {prompt}"
        )
        genai_response = self.genai_model.generate_content([prompt, image])
        return genai_response.text

    def handle_audio_processing(self, recognizer: sr.Recognizer, audio: sr.AudioData):
        """
        Callback function to process audio input once recognized.

        Args:
            recognizer (sr.Recognizer): The speech recognizer instance.
            audio (sr.AudioData): The audio data captured by the microphone.
        """

        data_folder_path = Path(os.path.abspath(os.path.join(".", "data")))
        os.makedirs(data_folder_path, exist_ok=True)

        audio_prompt_file_path = Path(
            os.path.abspath(os.path.join(data_folder_path, "user_audio_prompt.wav"))
        )
        with open(audio_prompt_file_path, "wb") as f:
            f.write(audio.get_wav_data())

        transcribed_text = self.transcribe_audio_to_text(
            audio_file_path=audio_prompt_file_path
        )
        parsed_prompt = self.extract_prompt(transcribed_text=transcribed_text)

        if parsed_prompt:
            utils.log_chat_message(
                log_file_path=self.log_file_path, user_message=parsed_prompt
            )
            skip_response = False

            selected_assistant_action = self.select_assistant_action(
                prompt=parsed_prompt
            )

            if "capture webcam" in selected_assistant_action:
                image_path = webcam.capture_webcam_image()
                image_analysis_result = self.analyze_image_prompt(
                    prompt=parsed_prompt, image_path=image_path
                )

            elif "take screenshot" in selected_assistant_action:
                image_path = utils.capture_screenshot()
                image_analysis_result = self.analyze_image_prompt(
                    prompt=parsed_prompt, image_path=image_path
                )

            elif "delete screenshot" in selected_assistant_action:
                utils.remove_last_screenshot()
                image_analysis_result = None
                ai_response = "Screenshot deleted successfully."
                self.text_to_speech(text=ai_response)

                utils.log_chat_message(
                    log_file_path=self.log_file_path, ai_message=ai_response
                )

                skip_response = True

            elif "extract clipboard" in selected_assistant_action:
                clipboard_content = utils.get_clipboard_text()
                parsed_prompt = (
                    f"{parsed_prompt}\n\nCLIPBOARD_CONTENT: {clipboard_content}"
                )
                image_analysis_result = None

            else:
                image_analysis_result = None

            # If the response is not supposed to be skipped, then generate the response and speak it out.
            if not skip_response:
                response = self.generate_chat_response_with_groq(
                    prompt=parsed_prompt, image_context=image_analysis_result
                )
                utils.log_chat_message(
                    log_file_path=self.log_file_path, ai_message=response
                )
                self.text_to_speech(text=response)

        # Remove the user prompt audio after the response is generated.
        if os.path.exists(audio_prompt_file_path):
            os.remove(audio_prompt_file_path)
