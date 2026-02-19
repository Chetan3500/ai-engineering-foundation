import os
import logging
import math
import hashlib
import time
from typing import Tuple
from google import genai
from collections import deque

logger = logging.getLogger(__name__)

request_timestamps = deque()
# ---------------------------
# Global User State (Multi-user simulation)
# ---------------------------
user_state = {}

circuit_state = {
    "failure_count": -1,
    "opened_at": None,
    "is_open": False
}

# ---------------------------
# Utility
# ---------------------------
def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def summarize_history(client, model_name, messages):
    summary_prompt = "Summarize the following conversation briefly but preserve key context:\n\n"

    for msg in messages:
        summary_prompt += f"{msg['role']}: {msg['content']}\n"

    response = client.models.generate_content(
        model=model_name,
        contents=summary_prompt,
        config={"max_output_tokens": 300}
    )

    return response.text if response and response.text else ""

cache_store = {}

def build_cache_key(prompt: str, model_name: str) -> str:
    raw_key = f"{model_name}:{prompt}"
    return hashlib.sha256(raw_key.encode()).hexdigest()

# ---------------------------
# Main LLM Call
# ---------------------------
def call_gemini(user_id: str, prompt: str) -> Tuple[bool, str]:

    # ---------- Initialize user ----------
    if user_id not in user_state:
        user_state[user_id] = {
            "memory": [],
            "request_timestamps": deque(),
            "token_usage": 0
        }

    user = user_state[user_id]

    # ---------- Per-user rate limiting ----------
    current_time = time.time()
    max_requests_per_minute = int(os.getenv("MAX_REQUESTS_PER_MINUTE", 10))

    while user["request_timestamps"] and current_time - user["request_timestamps"][0] > 60:
        user["request_timestamps"].popleft()

    if len(user["request_timestamps"]) >= max_requests_per_minute:
        logger.warning(f"Rate limit exceeded for user {user_id}")
        return False, "Too many requests. Slow down."

    user["request_timestamps"].append(current_time)

    # ---------- Config ----------
    api_key = os.getenv("GENAI_API_KEY")
    model_name = os.getenv("GENAI_MODEL_NAME", "gemini-flash-latest")
    fallback_model = os.getenv("FALLBACK_MODEL", "gemini-flash-lite-latest")

    if not api_key:
        return False, "API key not found"

    max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", 2000))
    max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", 1500))
    max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", 8500))
    max_memory_tokens = int(os.getenv("MAX_MEMORY_TOKENS", 3000))
    max_session_tokens = int(os.getenv("MAX_SESSION_TOKENS", 5000))

    cost_per_1k_tokens = float(os.getenv("COST_PER_1K_TOKENS", 0.0003))

    max_failures = int(os.getenv("CIRCUIT_MAX_FAILURES", 3))
    cooldown_seconds = int(os.getenv("CIRCUIT_COOLDOWN_SECONDS", 60))

    # ---------- Circuit breaker check ----------
    if circuit_state["is_open"]:
        if time.time() - circuit_state["opened_at"] < cooldown_seconds:
            return False, "LLM temporarily unavailable. Try later."
        else:
            logger.info("Circuit half-open. Testing recovery.")
            circuit_state["is_open"] = False
            circuit_state["failure_count"] = 0

    # ---------- Input token guard ----------
    input_tokens = estimate_tokens(prompt)
    if input_tokens > max_input_tokens:
        return False, "Input too large."

    client = genai.Client(api_key=api_key)

    try:
        # ---------- Append user message ----------
        user["memory"].append({"role": "user", "content": prompt})

        # ---------- Build full prompt ----------
        def build_prompt():
            fp = ""
            for msg in user["memory"]:
                fp += f"{msg['role']}: {msg['content']}\n"
            fp += "ASSISTANT:"
            return fp

        full_prompt = build_prompt()
        memory_tokens = estimate_tokens(full_prompt)

        # ---------- Memory summarization ----------
        if memory_tokens > max_memory_tokens:
            logger.warning("Memory exceeded. Summarizing...")

            old_messages = user["memory"][:-2]
            recent_messages = user["memory"][-2:]

            summary = summarize_history(client, model_name, old_messages)

            user["memory"].clear()
            user["memory"].append({
                "role": "system",
                "content": summary
            })
            user["memory"].extend(recent_messages)

            # REBUILD prompt after summarization
            full_prompt = build_prompt()
            memory_tokens = estimate_tokens(full_prompt)

        # ---------- Context window guard ----------
        if memory_tokens > max_context_tokens:
            return False, "Context window exceeded."

        # ---------- LLM Call with Retry ----------
        start_time = time.time()
    
        # ---------- Cache check ----------
        cache_ttl = int(os.getenv("CACHE_TTL_SECONDS", 300))

        cache_key = build_cache_key(prompt, model_name)
        current_time = time.time()

        if cache_key in cache_store:
            entry = cache_store[cache_key]
            if current_time - entry["timestamp"] < cache_ttl:
                logger.info("Cache HIT")
                return True, entry["response"]
            else:
                logger.info("Cache expired")
                del cache_store[cache_key]

        logger.info("Cache MISS")

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config={"max_output_tokens": max_output_tokens}
            )
            circuit_state["failure_count"] = 0

        except Exception as e:
            error_message = str(e)

            if "429" in error_message:
                logger.warning("429 detected. Retrying once...")
                time.sleep(2)

                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=full_prompt,
                        config={"max_output_tokens": max_output_tokens}
                    )
                    circuit_state["failure_count"] = 0

                except Exception:
                    logger.warning("Retry failed. Using fallback model...")
                    response = client.models.generate_content(
                        model=fallback_model,
                        contents=full_prompt,
                        config={"max_output_tokens": max_output_tokens}
                    )
                    circuit_state["failure_count"] = 0
            else:
                circuit_state["failure_count"] += 1
                response = None

        # ---------- Circuit open check ----------
        if circuit_state["failure_count"] >= max_failures:
            circuit_state["is_open"] = True
            circuit_state["opened_at"] = time.time()
            return False, "LLM disabled due to repeated failures."

        cache_store[cache_key] = {
            "response": response.text,
            "timestamp": current_time,
            "model": model_name
        }

        if not response or not response.text:
            return False, "Empty response"

        latency = time.time() - start_time
        logger.info(f"LLM latency: {latency:.3f}s")

        response_text = response.text

        # ---------- Append assistant message ----------
        user["memory"].append({"role": "assistant", "content": response_text})

        # ---------- Correct token accounting ----------
        prompt_tokens = estimate_tokens(full_prompt)
        output_tokens = estimate_tokens(response_text)
        total_tokens = prompt_tokens + output_tokens

        user["token_usage"] += total_tokens

        # ---------- Session quota ----------
        if user["token_usage"] > max_session_tokens:
            return False, "Session token limit reached."

        # ---------- Cost estimation ----------
        cost = (total_tokens / 1000) * cost_per_1k_tokens

        logger.info(f"Prompt tokens: {prompt_tokens}")
        logger.info(f"Output tokens: {output_tokens}")
        logger.info(f"Total tokens: {total_tokens}")
        logger.info(f"Estimated cost: ${cost:.6f}")

        context_usage = (total_tokens / max_context_tokens) * 100
        logger.info(f"Context usage: {context_usage:.2f}%")

        return True, response_text

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return False, str(e)
