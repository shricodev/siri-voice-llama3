import os
import sys

from dotenv import load_dotenv


def exit_program(status_code: int = 0) -> None:
    sys.exit(status_code)


def get_credentials():
    load_dotenv()

    # Get the username and password from the environment variables:
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    google_gen_ai_key: str | None = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    if any(key is None for key in (groq_api_key, google_gen_ai_key, openai_api_key)):
        exit_program(status_code=1)

    return groq_api_key, google_gen_ai_key, openai_api_key
