import os
import logging
from typing import Tuple

# import the google genai
from google import genai

logger = logging.getLogger(__name__)

def call_gemini(prompt: str) -> Tuple[bool, str]:
    api_key = os.getenv("GENAI_API_KEY")
    model_name = os.getenv("GENAI_MODEL_NAME", "gemini-2.5-flash")

    if not api_key:
        logger.error("API key not found")
        return False, "API key not found"

    client = genai.Client(api_key=api_key)

    logger.info(f"Sending prompt to Gemini model: {model_name}")
    logger.info(f"Prompt length: {len(prompt)} characters")

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        if not response:
            logger.error("Empty response from Gemini")
            return False, "Empty response"

        logger.info("Response received from Gemini")
        return True, response.text
    except Exception as e:
        logger.error("Gemini API error: %s", e)
        return False, str(e)