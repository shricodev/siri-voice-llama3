# Siri Voice LLAMA-3 🧙‍♂️🪄

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg?cacheSeconds=2592000)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)
[![Twitter: shricodev](https://img.shields.io/twitter/follow/shricodev.svg?style=social)](https://twitter.com/shricodev)

![GitHub repo size](https://img.shields.io/github/repo-size/shricodev/siri-voice-llama3?style=plastic)
![GitHub language count](https://img.shields.io/github/languages/count/shricodev/siri-voice-llama3?style=plastic)
![GitHub top language](https://img.shields.io/github/languages/top/shricodev/siri-voice-llama3?style=plastic)
![GitHub last commit](https://img.shields.io/github/last-commit/shricodev/siri-voice-llama3?color=red&style=plastic)

## 📚 Overview

The **AI Assistant Automation** is a Python application that uses **Llama3**, **gTTS**, **OpenAI**, **Groq**, and **Faster-Whisper** to create an intelligent assistant similar to **Siri**, with integrated image recognition support. This project allows users to interact with the assistant through voice commands and receive responses in audio format.

## 😎 Features

- **Voice Interaction**: Communicate with the assistant using voice prompts.
- **Audio Response**: The assistant responds with audio outputs generated by **gTTS/OpenAI/pyttsx3**.
- **Image Recognition**: Analyze and respond to images using advanced recognition techniques.
- **Chat History Logging**: Maintain a log of user interactions for better context and history tracking.

## ⚠️ Limitations

This application may have limitations based on the performance of the underlying AI models and available computing resources. Ensure that the necessary libraries are properly installed and the system is configured to handle audio and image processing efficiently.

## 🌳 Project Structure

```plaintext
siri-voice-llama3/
├── .git/
├── (gitignored) .venv/
├── data/
│   ├── ai_response/
│   │   └── .gitkeep
│   │   └── (gitignored) ai_response_audio.mp3
│   ├── chat_history/
│   │   └── 2024/
│   │      └── 10/
│   │         ├── (gitignored) 04.log
│   │         └── (gitignored) 05.log
│   └── .gitkeep
│   └── (gitignored) user_audio_prompt.wav
├── main.py
├── README.md
├── requirements.txt
└── src/
    ├── __pycache__/
    ├── setup.py
    ├── siri.py
    ├── utils.py
    └── webcam.py
```

## 🛠️ Installation

- **Clone the Repository**

> 💬 If you are using HTTPS protocol instead of SSH, change the `git clone` command accordingly.

```bash
git clone git@github.com:shricodev/siri-voice-llama3.git
cd siri-voice-llama3
```

- **Create and Activate Virtual Environment**

```bash
python3 -m venv .venv
source .venv/bin/activate.fish # or .venv/bin/activate if you are not using the fish shell
```

- **Install Dependencies**

```bash
pip install -r requirements.txt
```

- **Set Up Environment Variables**

```bash
GROQ_API_KEY=
GOOGLE_GENERATIVE_AI_API_KEY=

# Optional
OPENAI_API_KEY=
```

You can use the `.env.example` file as a template.

## 💻 Usage

- **Run the Assistant**

To start the assistant, execute the following command:

```bash
python main.py
```

This command initializes the assistant, allowing you to interact via voice commands.

## 💬 Logging

The application logs all interactions in the `data/chat_history/` directory. You can review past interactions in the log files to understand the context of your conversations.

## Show your support

Give a ⭐️ if this project helped you!
