import os
import logging
import math
import time
from typing import Tuple
from google import genai

logger = logging.getLogger(__name__)

def call_gemini(prompt: str) -> Tuple[bool, str]:
    api_key = os.getenv("GENAI_API_KEY")
    model_name = os.getenv("GENAI_MODEL_NAME", "gemini-2.5-flash")

    if not api_key:
        logger.error("API key not found")
        return False, "API key not found"

    max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", "200"))
    max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "500"))
    cost_per_1k_tokens = float(os.getenv("COST_PER_1K_TOKENS", "0.0003"))
    max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "1500"))
    
    input_tokens = estimate_tokens(prompt)
    logger.info(f"Estimated input tokens: {input_tokens}")
    
    if input_tokens > max_context_tokens:
        logger.warning("Input tokens exceed max context tokens limit.")
        return False, "Input exceeded max context tokens."

    if input_tokens > max_input_tokens:
        logger.warning("Input tokens exceed max tokens limit.")
        return False, "Input too large."

    client = genai.Client(api_key=api_key)

    # call the gemini api
    try:
        # start timing
        start_time = time.time()
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "max_output_tokens": max_output_tokens
            }
        )
        logger.info(f"Sending prompt to Gemini model: {model_name}")
        logger.info(f"Prompt length: {len(prompt)} characters")

        # end timing
        end_time = time.time()
        # latency
        latency = end_time - start_time
        logger.info(f"LLM Latency: {latency:.3f} seconds")

        if not response:
            logger.error("Empty response from Gemini")
            return False, "Empty response"
        
        response_text = response.text
        logger.info(f"Response length: {len(response_text)} characters")

        output_tokens = estimate_tokens(response_text)
        logger.info(f"Estimated output tokens: {output_tokens}")

        total_tokens = input_tokens + output_tokens
        logger.info(f"Estimated total tokens: {total_tokens}")

        cost = (total_tokens / 1000) * cost_per_1k_tokens
        logger.info(f"Estimated request cost: ${cost:.6f}")

        if total_tokens > max_context_tokens:
            logger.warning("Total tokens exceed max context tokens limit.")

        context_usage = (total_tokens / max_context_tokens) * 100
        logger.info(f"Context usage: {context_usage:.2f}%")

        return True, response_text

    except Exception as e:
        logger.error("Gemini API error: %s", e)
        return False, str(e)

# function estimate_tokens - returns the number of tokens in a string
def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)
