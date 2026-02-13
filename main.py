#!/usr/bin/env python

# import the logging module
import logging

# import the api_client module
from src.logger_config import setup_logger
from src.api_client import call_github_api

# setup the logger
setup_logger()
# get the logger
logger = logging.getLogger(__name__)

# define a main function to call the GitHub API and print the response
def main():
    # log the start of the application
    logger.info("Starting the application")

    status_code, response = call_github_api()

    # log the response
    if status_code is None:
        logger.error(f"API call failed with error: {response}")
    else:
        logger.info(f"API call successful with status code: {status_code}")
        logger.debug(f"Response Body: {response}")

    # log the end of the application
    logger.info("Finished the application")

# call the main function
if __name__ == "__main__":
    main()