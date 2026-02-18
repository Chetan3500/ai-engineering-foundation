import os
import logging
import math
import time
from typing import Tuple
from google import genai
from collections import deque

logger = logging.getLogger(__name__)

request_timestamps = deque()
user_state = {}

conversation_history = []

session_token_usage = 0

def summarize_history(client, model_name, messages):
    summary_prompt = "Summarize the following conversation briefly but preserve key context:\\n\\n"

    for msg in messages:
        summary_prompt += f"{msg['role']}: {msg['content']}\\n"

    response = client.models.generate_content(
        model=model_name,
        contents=summary_prompt,
        config={"max_output_tokens": 300}
    )

    return response.text if response and response.text else ""


def call_gemini(user_id: str, prompt: str) -> Tuple[bool, str]:
    if user_id not in user_state:
        user_state[user_id] = {
            "memory": [],
            "request_timestamps": deque(),
            "token_usage": 0
        }
    
    user = user_state[user_id]

    current_time = time.time()

    max_requests_per_minute = int(os.getenv("MAX_REQUESTS_PER_MINUTE", 10))

    # Remove timestamps older than 60 seconds
    while user["request_timestamps"] and current_time - user["request_timestamps"][0] > 60:
        user["request_timestamps"].popleft()

    if len(user["request_timestamps"]) >= max_requests_per_minute:
        logger.warning(f"Rate limit exceeded for user {user_id}")
        return False, "Too many requests. Slow down."

    user["request_timestamps"].append(current_time)

    api_key = os.getenv("GENAI_API_KEY")
    model_name = os.getenv("GENAI_MODEL_NAME", "gemini-2.5-flash")

    if not api_key:
        logger.error("API key not found")
        return False, "API key not found"

    max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", 2000))
    max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", 1500))
    cost_per_1k_tokens = float(os.getenv("COST_PER_1K_TOKENS", 0.0003))
    max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", 8500))
    max_memory_tokens = int(os.getenv("MAX_MEMORY_TOKENS", 3000))
    
    input_tokens = estimate_tokens(prompt)
    logger.info(f"Estimated input tokens: {input_tokens}")
    
    if input_tokens > max_context_tokens:
        logger.warning("Input tokens exceed max context tokens limit.")
        return False, "Input exceeded max context tokens."

    if input_tokens > max_input_tokens:
        logger.warning("Input tokens exceed max tokens limit.")
        return False, "Input too large."

    client = genai.Client(api_key=api_key)

    try:
        user["memory"].append({
            "role": "user",
            "content": prompt
        })

        full_prompt = ""
        for message in user["memory"]:
            full_prompt += f"{message['role']}: {message['content']}\n"
        full_prompt += "ASSISTANT: "

        memory_prompt_tokens = estimate_tokens(full_prompt)
        logger.info(f"Estimated memory prompt tokens: {memory_prompt_tokens}")

        if memory_prompt_tokens > max_memory_tokens:
            logger.warning("Memory exceed limit, summarizing old messages")
            
            old_messages = user["memory"][-2:]
            recent_messages = user["memory"][:-2]

            summary = summarize_history(client, model_name, old_messages)
            
            # replace old messages with summary and recent messages
            user["memory"].clear()
            user["memory"].append({
                "role": "system",
                "content": summary
            })
            user["memory"].extend(recent_messages)

        start_time = time.time()

        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
            config={
                "max_output_tokens": max_output_tokens
            }
        )
        logger.info(f"Sending prompt to Gemini model: {model_name}")
        logger.info(f"Prompt length: {len(full_prompt)} characters")

        end_time = time.time()
        latency = end_time - start_time
        logger.info(f"LLM Latency: {latency:.3f} seconds")

        if not response:
            logger.error("Empty response from Gemini")
            return False, "Empty response"
        
        response_text = response.text
        logger.info(f"Response length: {len(response_text)} characters")

        user["memory"].append({
            "role": "assistant",
            "content": response_text
        })

        output_tokens = estimate_tokens(response_text)
        logger.info(f"Estimated output tokens: {output_tokens}")

        total_tokens = input_tokens + output_tokens
        logger.info(f"Estimated total tokens: {total_tokens}")

        user["token_usage"] += total_tokens

        cost = (total_tokens / 1000) * cost_per_1k_tokens
        logger.info(f"Estimated request cost: ${cost:.6f}")

        if total_tokens > max_context_tokens:
            logger.warning("Total tokens exceed max context tokens limit.")

        context_usage = (total_tokens / max_context_tokens) * 100
        logger.info(f"Context usage: {context_usage:.2f}%")

        max_session_tokens = int(os.getenv("MAX_SESSION_TOKENS", 5000))

        user["token_usage"] += total_tokens

        logger.info(f"Session token usage: {user['token_usage']}")

        if user['token_usage'] > max_session_tokens:
            logger.warning(f"Session token quota exceeded for user {user_id}")
            return False, "Session token limit reached."

        return True, response_text

    except Exception as e:
        logger.error("Gemini API error: %s", e)
        return False, str(e)

# function estimate_tokens - returns the number of tokens in a string
def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)
