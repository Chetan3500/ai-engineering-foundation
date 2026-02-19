import os
import time
import math
import hashlib
import logging
from typing import Dict, List, Tuple
from collections import deque
from google import genai

logger = logging.getLogger(__name__)

# ==============================
# Global In-Memory Structures
# ==============================

# Per-user state store
users_state: Dict[str, dict] = {}

# Global cache (safe, model-aware)
cache_store: Dict[str, dict] = {}

# ==============================
# Utility Functions
# ==============================

def estimate_tokens(text: str) -> int:
    # Rough estimation: 1 token ≈ 4 characters
    return math.ceil(len(text) / 4)


def build_cache_key(prompt: str, model_name: str) -> str:
    raw_key = f"{model_name}:{prompt}"
    return hashlib.sha256(raw_key.encode()).hexdigest()


# ==============================
# LLM Service
# ==============================

class LLMService:

    def __init__(self):
        self.api_key = os.getenv("GENAI_API_KEY")
        self.model_name = os.getenv("GENAI_MODEL_NAME", "gemini-2.5-flash")

        self.max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", "1000"))
        self.max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "500"))
        self.max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "8000"))
        self.max_memory_tokens = int(os.getenv("MAX_MEMORY_TOKENS", "3000"))
        self.max_session_tokens = int(os.getenv("MAX_SESSION_TOKENS", "5000"))
        self.max_requests_per_minute = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "10"))

        self.cost_per_1k_tokens = float(os.getenv("COST_PER_1K_TOKENS", "0.0003"))
        self.cache_ttl = int(os.getenv("CACHE_TTL_SECONDS", "300"))

        if not self.api_key:
            raise ValueError("GENAI_API_KEY not set")

        self.client = genai.Client(api_key=self.api_key)

    # ------------------------------
    # User State Management
    # ------------------------------

    def _get_user_state(self, user_id: str) -> dict:
        if user_id not in users_state:
            users_state[user_id] = {
                "conversation_history": [],
                "token_usage": 0,
                "request_timestamps": deque()
            }
        return users_state[user_id]

    # ------------------------------
    # Rate Limiting
    # ------------------------------

    def _check_rate_limit(self, user_state: dict) -> bool:
        now = time.time()

        while user_state["request_timestamps"] and now - user_state["request_timestamps"][0] > 60:
            user_state["request_timestamps"].popleft()

        if len(user_state["request_timestamps"]) >= self.max_requests_per_minute:
            return False

        user_state["request_timestamps"].append(now)
        return True

    # ------------------------------
    # Summarization
    # ------------------------------

    def _summarize_history(self, messages: List[dict]) -> str:
        summary_prompt = "Summarize this conversation briefly, preserving key context:\n\n"
        for msg in messages:
            summary_prompt += f"{msg['role']}: {msg['content']}\n"

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=summary_prompt,
            config={"max_output_tokens": 300}
        )

        return response.text if response and response.text else ""

    # ------------------------------
    # Chat Method
    # ------------------------------

    def chat(self, user_id: str, prompt: str) -> Tuple[bool, str]:

        user_state = self._get_user_state(user_id)

        # ---- Rate Limit Check ----
        if not self._check_rate_limit(user_state):
            logger.warning(f"Rate limit exceeded for user {user_id}")
            return False, "Too many requests. Please slow down."

        # ---- Token Guard ----
        input_tokens = estimate_tokens(prompt)
        logger.info(f"[{user_id}] Input tokens: {input_tokens}")

        if input_tokens > self.max_input_tokens:
            return False, "Input exceeds maximum token limit."

        # ---- Cache Check (Stateless Only) ----
        cache_key = build_cache_key(prompt, self.model_name)
        now = time.time()

        if cache_key in cache_store:
            entry = cache_store[cache_key]
            if now - entry["timestamp"] < self.cache_ttl:
                logger.info(f"[{user_id}] Cache HIT")
                return True, entry["response"]
            else:
                del cache_store[cache_key]

        logger.info(f"[{user_id}] Cache MISS")

        # ---- Append to Memory ----
        user_state["conversation_history"].append({
            "role": "user",
            "content": prompt
        })

        # ---- Build Context ----
        full_prompt = ""
        for msg in user_state["conversation_history"]:
            full_prompt += f"{msg['role']}: {msg['content']}\n"
        full_prompt += "assistant: "

        memory_tokens = estimate_tokens(full_prompt)

        # ---- Memory Compression ----
        if memory_tokens > self.max_memory_tokens:
            logger.warning(f"[{user_id}] Memory limit exceeded, summarizing")

            old_messages = user_state["conversation_history"][:-2]
            recent_messages = user_state["conversation_history"][-2:]

            summary = self._summarize_history(old_messages)

            user_state["conversation_history"] = [{
                "role": "system",
                "content": f"Conversation summary: {summary}"
            }] + recent_messages

        # ---- LLM Call ----
        start_time = time.time()

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config={"max_output_tokens": self.max_output_tokens}
        )

        latency = time.time() - start_time
        logger.info(f"[{user_id}] LLM latency: {latency:.3f}s")

        if not response or not response.text:
            return False, "Empty response from LLM."

        response_text = response.text

        # ---- Update Memory ----
        user_state["conversation_history"].append({
            "role": "assistant",
            "content": response_text
        })

        # ---- Token Accounting ----
        output_tokens = estimate_tokens(response_text)
        total_tokens = input_tokens + output_tokens

        user_state["token_usage"] += total_tokens

        logger.info(f"[{user_id}] Total tokens: {total_tokens}")
        logger.info(f"[{user_id}] Session tokens: {user_state['token_usage']}")

        # ---- Session Quota ----
        if user_state["token_usage"] > self.max_session_tokens:
            return False, "Session token limit exceeded."

        # ---- Cost Estimation ----
        estimated_cost = (total_tokens / 1000) * self.cost_per_1k_tokens
        logger.info(f"[{user_id}] Estimated cost: ${estimated_cost:.6f}")

        # ---- Context Usage ----
        context_usage = (total_tokens / self.max_context_tokens) * 100
        logger.info(f"[{user_id}] Context usage: {context_usage:.2f}%")

        # ---- Store in Cache ----
        cache_store[cache_key] = {
            "response": response_text,
            "timestamp": now,
            "model": self.model_name
        }

        return True, response_text
