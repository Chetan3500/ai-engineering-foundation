import requests

# define a function to call the GitHub API 
# return the status code and response body
def call_github_api():
    # The request will time out if the response takes longer than 5 seconds.
    try:    
        response = requests.get("https://api.github.com", timeout=5)
        # return the status code and response body
        return response.status_code, response.json()
    # Handle timeout
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    # Handle HTTP errors
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP error: {e}"
    # Handle other request exceptions
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {e}"