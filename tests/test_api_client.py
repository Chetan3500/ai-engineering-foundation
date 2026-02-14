from src.api_client import call_github_api

def test_api_returns_status_or_error():
    status_code, response = call_github_api()
    assert status_code is None or isinstance(status_code, int)
    assert isinstance(response, dict) or isinstance(response, str)