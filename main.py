#!/usr/bin/env python

import logging
import argparse
from src.logger_config import setup_logger
from src.api_client import call_github_api
from src.llm_client import call_gemini

setup_logger()
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Github API CLI Tool")
    parser.add_argument("--show-data", action="store_true", help="Display API response data")
    parser.add_argument("--user-id", type=str, help="User ID for LLM")
    parser.add_argument("--prompt", type=str, help="Prompt for LLM")
    args = parser.parse_args()

    if args.show_data:
        logger.info("Starting the application")

        status_code, response = call_github_api()

        if status_code:
            logger.info(f"API call successful with status code: {status_code}")
            if args.show_data:
                logger.debug("Response Body: %s", response)
        else:
            logger.error("API call failed with error: %s", response)

        logger.info("Finished the application")
    elif args.user_id or args.prompt:
        if not args.prompt:
            args.prompt = input("Enter a prompt: ")
        
        while True:
            if args.user_id:
                success, result = call_gemini(user_id=args.user_id, prompt=args.prompt)
            else:
                success, result = call_gemini(user_id=None, prompt=args.prompt)
            if success:
                logger.info("LLM call successful with result: %s", result)
            else:
                logger.error("LLM call failed with error: %s", result)
            
            args.prompt = input("Enter a prompt (or 'exit' to quit): ")
            if args.prompt.lower() == "exit":
                break
                        
    else:
        logger.error("No arguments provided. Use --show-data or --user-id and --prompt")

if __name__ == "__main__":
    main()