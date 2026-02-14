import requests
import logging
import os

# import typing for Tuple and Union
# Tuple: a collection of values of the same type
# Union: a collection of values of different types
from typing import Tuple, Union

# get the logger
# __name__ is the name of the module
logger = logging.getLogger(__name__)

# define a function to call the GitHub API 
# return the status code and response body
def call_github_api() -> Tuple[Union[int, None], Union[dict, str]]:
    # The request will time out if the response takes longer than 5 seconds.
    timeout = int(os.getenv("API_TIMEOUT", 5))
    try:
        logger.info(f"Calling Github API with timeout: {timeout}s")

        response = requests.get("https://api.github.com", timeout=timeout)
        response.raise_for_status()

        logger.info("Github API call successful")
        return response.status_code, response.json()
    # Handle timeout
    except requests.exceptions.Timeout as e:
        logger.error("API call timed out: %s", e)
        return None, "Request timed out"
    # Handle HTTP errors
    except requests.exceptions.HTTPError as e:
        logger.error("API call failed with HTTP error: %s", e)
        return None, f"HTTP error: {e}"
    # Handle other request exceptions
    except requests.exceptions.RequestException as e:
        logger.error("API call failed: %s", e)
        return None, f"Request failed: {e}"