#!/usr/bin/env python

import logging
import argparse
from src.llm_client import LLMService

# -----------------------------
# Logging Setup
# -----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[ %(levelname)s - %(name)s - %(asctime)s ] %(message)s",
)

logger = logging.getLogger(__name__)


def run_cli():
    parser = argparse.ArgumentParser(description="LLM CLI Interface")
    parser.add_argument("--user-id", type=str, required=True, help="User ID")
    args = parser.parse_args()

    user_id = args.user_id

    service = LLMService()

    logger.info(f"Starting CLI for user: {user_id}")

    while True:
        prompt = input("You: ")

        if prompt.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break

        success, response = service.chat(user_id=user_id, prompt=prompt)

        if success:
            print(f"AI: {response}\n")
        else:
            print(f"Error: {response}\n")


if __name__ == "__main__":
    run_cli()