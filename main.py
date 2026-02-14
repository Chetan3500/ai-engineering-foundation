#!/usr/bin/env python

# import the modules
import logging
import argparse

# import the api_client module
from src.logger_config import setup_logger
from src.api_client import call_github_api

# import the llm_client module
from src.llm_client import call_gemini

# setup the logger
setup_logger()
# get the logger
logger = logging.getLogger(__name__)

# define a main function to call the GitHub API and print the response
def main():
    parser = argparse.ArgumentParser(description="Github API CLI Tool")
    parser.add_argument("--show-data", action="store_true", help="Display API response data")
    parser.add_argument("--prompt", type=str, help="Prompt for LLM")
    args = parser.parse_args()

    logger.info("Starting the application")

    status_code, response = call_github_api()

    if status_code:
        logger.info(f"API call successful with status code: {status_code}")
        if args.show_data:
            logger.debug("Response Body: %s", response)
    else:
        logger.error("API call failed with error: %s", response)

    logger.info("Finished the application")

    # if prompt is provided then call LLM and print the response or error
    if args.prompt:
        success, result = call_gemini(args.prompt)
        if success:
            logger.info("LLM call successful with result: %s", result)
        else:
            logger.error("LLM call failed with error: %s", result)
    

# call the main function
if __name__ == "__main__":
    main()