
import re
import requests

request_headers = {
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
}

def get_plain_content_from(url: str) -> str:
    return requests.get(url, headers=request_headers).text

def regex_replace_chars_not_between_a_and_z_with(input: str, replacement: str) -> str:
    return re.sub(r"[^a-zA-Z]", replacement, input)

def create_simple_identifier_from_url(url: str) -> str:
    return regex_replace_chars_not_between_a_and_z_with(url, "_")
