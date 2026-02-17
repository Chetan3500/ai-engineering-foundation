import os
import logging
import math
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

    # read environment varaible
    # MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS and COST_PER_1K_TOKENS
    # with default value 1000, 500 & 0.0003 respectively
    max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", "1000"))
    max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "500"))
    cost_per_1k_tokens = float(os.getenv("COST_PER_1K_TOKENS", "0.0003"))
    
    input_tokens = estimate_tokens(prompt)

    logger.info(f"Estimated input tokens: {input_tokens}")
    
    # check if input tokens exceed max input tokens
    if input_tokens > max_input_tokens:
        logger.warning("Input tokens exceed max tokens limit.")
        return False, "Input too large."

    # create a client using the api key
    client = genai.Client(api_key=api_key)

    logger.info(f"Sending prompt to Gemini model: {model_name}")
    logger.info(f"Prompt length: {len(prompt)} characters")

    # call the gemini api
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "max_output_tokens": max_output_tokens
            }
        )

        # check if response is empty
        if not response:
            logger.error("Empty response from Gemini")
            return False, "Empty response"
        
        response_text = response.text

        # estimate output tokens
        output_tokens = estimate_tokens(response_text)
        # calculate total tokens
        total_tokens = input_tokens + output_tokens

        # calculate cost
        cost = (total_tokens / 1000) * cost_per_1k_tokens

        # log the results
        logger.info(f"Estimated output tokens: {output_tokens}")
        logger.info(f"Estimated total tokens: {total_tokens}")
        logger.info(f"Estimated request cost: ${cost:.6f}")

        # log the response length
        logger.info(f"Response length: {len(response_text)} characters")
        
        return True, response_text
    except Exception as e:
        logger.error("Gemini API error: %s", e)
        return False, str(e)

# function estimate_tokens - returns the number of tokens in a string
def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)
