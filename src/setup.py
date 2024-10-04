import os

from dotenv import load_dotenv

import utils


def get_credentials() -> tuple[str, str, str]:
    """
    Load API keys from environment variables and return them as a tuple.

    This function loads environment variables from a `.env` file using `dotenv`.
    It retrieves the Groq API key, Google Generative AI API key, and OpenAI API key.
    If any of the keys are missing, it exits the program with an error message.

    Returns:
        tuple[str, str, str]: A tuple containing the Groq API key, Google Generative AI API key,
                              and OpenAI API key.

    Raises:
        SystemExit: If any of the required API keys are not found, the program exits with an error message.
    """
    load_dotenv()

    # Get the username and password from the environment variables:
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    google_gen_ai_api_key: str | None = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    if groq_api_key is None or google_gen_ai_api_key is None or openai_api_key is None:
        return utils.exit_program(
            status_code=1,
            message="Missing API key(s). Make sure to set all of them in `.env` file.",
        )

    return groq_api_key, google_gen_ai_api_key, openai_api_key
