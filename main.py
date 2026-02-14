#!/usr/bin/env python

# import the modules
import logging
import argparse

# import the api_client module
from src.logger_config import setup_logger
from src.api_client import call_github_api

# setup the logger
setup_logger()
# get the logger
logger = logging.getLogger(__name__)

# define a main function to call the GitHub API and print the response
def main():
    parser = argparse.ArgumentParser(description="Github API CLI Tool")
    parser.add_argument("--show-data", action="store_true", help="Display API response data")
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

# call the main function
if __name__ == "__main__":
    main()