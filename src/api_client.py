import requests

# define a function to call the GitHub API 
# return the status code and response body
def call_github_api():
    response = requests.get("https://api.github.com")
    return response.status_code, response.json()