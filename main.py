#!/usr/bin/env python

# import the api_client module
from src.api_client import call_github_api

# define a main function to call the GitHub API and print the response
def main():
    status_code, response = call_github_api()
    print("Status Code:", status_code)
    print("Response Body:", response)

# call the main function
if __name__ == "__main__":
    main()