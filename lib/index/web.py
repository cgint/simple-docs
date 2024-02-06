
import re
import requests
import os
import hashlib

html_dl_cache_dir = "/data/html_dl_cache"
ignore_html_dl_cache = os.environ.get("IGNORE_HTML_DL_CACHE", "false").lower() == "true"

request_headers = {
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
}

def get_cache_file_from_url(url: str) -> str:
    url_sha256 = hashlib.sha256(url.encode()).hexdigest()
    domain_from_url = url.split("/")[2]
    path_part = create_simple_identifier_from_url(domain_from_url)
    file_prefix = create_simple_identifier_from_url(url[-30:])
    cache_dir = f"{html_dl_cache_dir}/{path_part}"
    cache_file = f"{cache_dir}/{url_sha256}_{file_prefix}"
    return cache_dir, cache_file

def get_plain_content_from_url_or_cache(url: str) -> str:
    cache_dir, cache_file = get_cache_file_from_url(url)
    os.makedirs(cache_dir, exist_ok=True)

    if not ignore_html_dl_cache and os.path.exists(cache_file):
        print(f"Using cached content from {cache_file} ...")
        with open(cache_file, "r") as f:
            return f.read()
    url_text = requests.get(url, headers=request_headers).text
    # always write to cache - so it can be used next time when not ignoring cache
    with open(cache_file, "w") as f:
        f.write(url_text)
    return url_text

def get_plain_content_from(url: str) -> str:
    return get_plain_content_from_url_or_cache(url)

def regex_replace_chars_not_between_a_and_z_with(input: str, replacement: str) -> str:
    return re.sub(r"[^a-zA-Z]", replacement, input)

def create_simple_identifier_from_url(url: str) -> str:
    return regex_replace_chars_not_between_a_and_z_with(url, "_")
