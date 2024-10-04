import os
import sys

# Add the src directory to the module search path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src import setup, siri

"""
Main entry point for the AI llama3 siri voice assitant.

This script loads the necessary API credentials from environment variables,
initializes the Siri assistant with the provided keys, and starts listening
for user input. The program will exit if any of the required API keys are
missing.

To run the application, execute this script in an environment where the
`.env` file is properly configured with the required API keys.
"""

if __name__ == "__main__":
    all_api_keys = setup.get_credentials()
    groq_api_key, google_gen_ai_api_key, openai_api_key = all_api_keys

    siri = siri.Siri(
        groq_api_key=groq_api_key,
        google_gen_ai_api_key=google_gen_ai_api_key,
        openai_api_key=openai_api_key,
    )

    siri.start_listening()
