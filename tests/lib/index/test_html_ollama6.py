import pytest
from unittest.mock import Mock, patch
import json
from typing import List
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from llama_index.core import Document
from lib.index.web import create_simple_identifier_from_url, get_plain_content_from
from bs4 import BeautifulSoup
from lib.index.html import (get_documents_from_urls, extract_content_part_from_html, clean_html_content, get_urls_from_html_content, create_doc_from_plain_html_content, get_documents_from_urls_as_mirror, get_documents_from_urls_as_mirror_rec)

# METHOD:
# def get_documents_from_urls(urls: List[str], producer_sink=lambda: Document) -> List[Document]:
# def test_get_documents_from_valid_urls():
#     urls = ["http://example.com", "https://example.org"]
#     producer_sink = Mock()
#     documents = get_documents_from_urls(urls, producer_sink)
#     assert len(documents) == 2
#     for document in documents:
#         assert isinstance(document, Document)

# def test_get_documents_from_invalid_urls():
#     urls = ["http://example.com", "https://example.org", "not-a-url"]
#     producer_sink = Mock()
#     documents = get_documents_from_urls(urls, producer_sink)
#     assert len(documents) == 2
#     for document in documents:
#         assert isinstance(document, Document)

# def test_get_documents_from_empty_list():
#     urls = []
#     producer_sink = Mock()
#     documents = get_documents_from_urls(urls, producer_sink)
#     assert len(documents) == 0

# def test_get_documents_from_single_url():
#     url = "http://example.com"
#     producer_sink = Mock()
#     documents = get_documents_from_urls([url], producer_sink)
#     assert len(documents) == 1
#     for document in documents:
#         assert isinstance(document, Document)

# def test_get_documents_from_mixed_urls():
#     urls = ["http://example.com", "https://example.org", "not-a-url"]
#     producer_sink = Mock()
#     documents = get_documents_from_urls(urls, producer_sink)
#     assert len(documents) == 2
#     for document in documents:
#         assert isinstance(document, Document)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_documents_from_urls` method:
# 
# 1. Test with valid URLs:
# ```python
# def test_get_documents_from_valid_urls():
#     urls = ["http://example.com", "https://example.org"]
#     producer_sink = Mock()
#     documents = get_documents_from_urls(urls, producer_sink)
#     assert len(documents) == 2
#     for document in documents:
#         assert isinstance(document, Document)
# ```
# This test case checks that the `get_documents_from_urls` method returns a list of two `Document` objects when given a list of valid URLs.
# 
# 2. Test with invalid URLs:
# ```python
# def test_get_documents_from_invalid_urls():
#     urls = ["http://example.com", "https://example.org", "not-a-url"]
#     producer_sink = Mock()
#     documents = get_documents_from_urls(urls, producer_sink)
#     assert len(documents) == 2
#     for document in documents:
#         assert isinstance(document, Document)
# ```
# This test case checks that the `get_documents_from_urls` method returns a list of two `Document` objects when given a list of valid URLs and one invalid URL.
# 
# 3. Test with empty list of URLs:
# ```python
# def test_get_documents_from_empty_list():
#     urls = []
#     producer_sink = Mock()
#     documents = get_documents_from_urls(urls, producer_sink)
#     assert len(documents) == 0
# ```
# This test case checks that the `get_documents_from_urls` method returns an empty list when given an empty list of URLs.
# 
# 4. Test with a single URL:
# ```python
# def test_get_documents_from_single_url():
#     url = "http://example.com"
#     producer_sink = Mock()
#     documents = get_documents_from_urls([url], producer_sink)
#     assert len(documents) == 1
#     for document in documents:
#         assert isinstance(document, Document)
# ```
# This test case checks that the `get_documents_from_urls` method returns a list of one `Document` object when given a single valid URL.
# 
# 5. Test with a mix of valid and invalid URLs:
# ```python
# def test_get_documents_from_mixed_urls():
#     urls = ["http://example.com", "https://example.org", "not-a-url"]
#     producer_sink = Mock()
#     documents = get_documents_from_urls(urls, producer_sink)
#     assert len(documents) == 2
#     for document in documents:
#         assert isinstance(document, Document)
# ```
# This test case checks that the `get_documents_from_urls` method returns a list of two `Document` objects when given a mix of valid and invalid URLs.



# METHOD:
# def extract_content_part_from_html(plain_html_content: str, initial_tag: str = "body") -> BeautifulSoup:
def test_extract_content_part_from_html():
    html = "<html><body><article>Test content</article></body></html>"
    expected_result = "<body><article>Test content</article></body>"
    assert str(extract_content_part_from_html(html)) == expected_result

def test_extract_content_part_from_html_with_initial_tag():
    html = "<html><body><div>Test content</div></body></html>"
    initial_tag = "div"
    expected_result = "<div>Test content</div>"
    assert str(extract_content_part_from_html(html, initial_tag)) == expected_result

def test_extract_content_part_from_html_with_invalid_initial_tag():
    html = "<html><body><article>Test content</article></body></html>"
    initial_tag = "invalid"
    expected_result = "<body><article>Test content</article></body>"
    assert str(extract_content_part_from_html(html, initial_tag)) == expected_result

# FULL-RESPONSE:
# # assistant: 
# Here are the test methods for the `extract_content_part_from_html` method:
# ```python
# def test_extract_content_part_from_html():
#     html = "<html><body><article>Test content</article></body></html>"
#     expected_result = "<body><article>Test content</article></body>"
#     assert str(extract_content_part_from_html(html) == expected_result
# 
# def test_extract_content_part_from_html_with_initial_tag():
#     html = "<html><body><div>Test content</div></body></html>"
#     initial_tag = "div"
#     expected_result = "<div>Test content</div>"
#     assert str(extract_content_part_from_html(html, initial_tag) == expected_result
# 
# def test_extract_content_part_from_html_with_invalid_initial_tag():
#     html = "<html><body><article>Test content</article></body></html>"
#     initial_tag = "invalid"
#     expected_result = "<body><article>Test content</article></body>"
#     assert str(extract_content_part_from_html(html, initial_tag) == expected_result
# ```



# METHOD:
# def clean_html_content(plain_html_content: str) -> str:
def test_clean_html_content_empty():
    assert clean_html_content("") == ""

def test_clean_html_content_single_article():
    html = "<html><body><article>Test content</article></body></html>"
    assert clean_html_content(html) == "Test content"

def test_clean_html_content_multiple_articles():
    html = "<html><body><article>Test content</article><article>Another test</article></body></html>"
    assert clean_html_content(html) == "Test content"

def test_clean_html_content_no_articles():
    html = "<html><body>No articles</body></html>"
    assert clean_html_content(html) == "No articles"

def test_clean_html_content_invalid():
    html = "<html><body>Invalid HTML"
    assert clean_html_content(html) == "Invalid HTML"

def test_test_clean_html_content_table():
    html = """
<html><body>
<table>
 <tr><td>Cell11</td><td>Cell12</td></tr>
 <tr><td>Cell21</td><td>Cell22</td></tr>
 </table>
 </body></html>"""
    assert clean_html_content(html) == "\n\nCell11Cell12\nCell21Cell22\n\n"

# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `clean_html_content` method:
# 
# 1. Test that the method returns an empty string when the input HTML content is empty:
# ```python
# def test_clean_html_content_empty():
#     assert clean_html_content("") == ""
# ```
# 2. Test that the method returns the expected text when the input HTML content contains a single "article" tag:
# ```python
# def test_clean_html_content_single_article():
#     html = "<html><body><article>Test content</article></body></html>"
#     assert clean_html_content(html) == "Test content"
# ```
# 3. Test that the method returns the expected text when the input HTML content contains multiple "article" tags:
# ```python
# def test_clean_html_content_multiple_articles():
#     html = "<html><body><article>Test content</article><article>Another test</article></body></html>"
#     assert clean_html_content(html) == "Test content\nAnother test"
# ```
# 4. Test that the method returns an empty string when the input HTML content does not contain any "article" tags:
# ```python
# def test_clean_html_content_no_articles():
#     html = "<html><body>No articles</body></html>"
#     assert clean_html_content(html) == ""
# ```
# 5. Test that the method returns an empty string when the input HTML content is not valid:
# ```python
# def test_clean_html_content_invalid():
#     html = "<html><body>Invalid HTML"
#     assert clean_html_content(html) == ""
# ```



# METHOD:
# def get_urls_from_html_content(plain_html_content: str) -> List[str]:
def test_no_urls():
    assert get_urls_from_html_content("<html><body>helloo</body></html>") == []

def test_with_urls():
    urls = ["http://example.com", "https://example.org"]
    html_content = f"<html><body>{get_a_href_tags(urls)}</body></html>"
    assert get_urls_from_html_content(html_content) == urls

def test_invalid_html():
    html_content = "<html><body>Invalid HTML"
    assert get_urls_from_html_content(html_content) == []

def test_missing_href():
    html_content = "<html><body><a>Invalid href</a></body></html>"
    assert get_urls_from_html_content(html_content) == []

def get_a_href_tags(urls) -> str:
    a_href_tags = [f"<a href='{url}'>Link</a>" for url in urls]
    return " any text really and a tag <br/>".join(a_href_tags)

def test_multiple_anchor_tags():
    urls = ["http://example.com", "https://example.org"]
    html_content = f"<html><body>{get_a_href_tags(urls)}</body></html>"
    assert get_urls_from_html_content(html_content) == urls


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_urls_from_html_content` method:
# 
# 1. Test that the method returns an empty list when there are no URLs in the HTML content:
# ```python
# def test_no_urls(plain_html_content):
#     assert get_urls_from_html_content(plain_html_content) == []
# ```
# 2. Test that the method returns a list of URLs when there are URLs in the HTML content:
# ```python
# def test_with_urls(plain_html_content):
#     urls = ["http://example.com", "https://example.org"]
#     html_content = f"<html><body>{urls[0]}</body></html>"
#     assert get_urls_from_html_content(html_content) == urls
# ```
# 3. Test that the method handles invalid HTML content:
# ```python
# def test_invalid_html(plain_html_content):
#     html_content = "<html><body>Invalid HTML"
#     assert get_urls_from_html_content(html_content) == []
# ```
# 4. Test that the method handles missing `href` attribute in anchor tags:
# ```python
# def test_missing_href(plain_html_content):
#     html_content = "<html><body><a>Invalid href</a></body></html>"
#     assert get_urls_from_html_content(html_content) == []
# ```
# 5. Test that the method handles multiple anchor tags with `href` attribute:
# ```python
# def test_multiple_anchor_tags(plain_html_content):
#     urls = ["http://example.com", "https://example.org"]
#     html_content = f"<html><body>{urls[0]}</body></html>"
#     assert get_urls_from_html_content(html_content) == urls
# ```



# METHOD:
# def create_doc_from_plain_html_content(url, plain_html_content: str, mirror_base=None) -> Document:
def test_create_doc_from_plain_html_content():
    url = "http://example.com"
    plain_html_content = "<html><body>Test content</body></html>"
    expected_text = "Test content"
    expected_metadata = {
        "source_id": url,
        "source_type": "html",
        "simple_id": create_simple_identifier_from_url(url)
    }
    document = create_doc_from_plain_html_content(url, plain_html_content)
    assert document.text == expected_text
    assert document.metadata == expected_metadata

def test_create_doc_from_plain_html_content_with_mirror_base():
    url = "http://example.com"
    plain_html_content = "<html><body>Test content</body></html>"
    mirror_base = "http://example.com/mirror/"
    expected_text = "Test content"
    expected_metadata = {
        "source_id": url,
        "source_type": "html",
        "simple_id": create_simple_identifier_from_url(url),
        "mirror_base": mirror_base
    }
    document = create_doc_from_plain_html_content(url, plain_html_content, mirror_base=mirror_base)
    assert document.text == expected_text
    assert document.metadata == expected_metadata

def test_create_doc_from_plain_html_content_without_mirror_base():
    url = "http://example.com"
    plain_html_content = "<html><body>Test content</body></html>"
    expected_text = "Test content"
    expected_metadata = {
        "source_id": url,
        "source_type": "html",
        "simple_id": create_simple_identifier_from_url(url)
    }
    document = create_doc_from_plain_html_content(url, plain_html_content)
    assert document.text == expected_text
    assert document.metadata == expected_metadata


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `create_doc_from_plain_html_content` method:
# 
# 1. Test that the returned document has the correct text and metadata:
# ```python
# def test_create_doc_from_plain_html_content():
#     url = "http://example.com"
#     plain_html_content = "<html><body>Test content</body></html>"
#     expected_text = "Test content"
#     expected_metadata = {
#         "source_id": url,
#         "source_type": "html",
#         "simple_id": create_simple_identifier_from_url(url)
#     }
#     if mirror_base is not None:
#         expected_metadata["mirror_base"] = mirror_base
#     document = create_doc_from_plain_html_content(url, plain_html_content)
#     assert document.text == expected_text
#     assert document.metadata == expected_metadata
# ```
# 2. Test that the returned document has the correct text and metadata when `mirror_base` is specified:
# ```python
# def test_create_doc_from_plain_html_content_with_mirror_base():
#     url = "http://example.com"
#     plain_html_content = "<html><body>Test content</body></html>"
#     mirror_base = "http://example.com/mirror/"
#     expected_text = "Test content"
#     expected_metadata = {
#         "source_id": url,
#         "source_type": "html",
#         "simple_id": create_simple_identifier_from_url(url),
#         "mirror_base": mirror_base
#     }
#     document = create_doc_from_plain_html_content(url, plain_html_content, mirror_base=mirror_base)
#     assert document.text == expected_text
#     assert document.metadata == expected_metadata
# ```
# 3. Test that the returned document has the correct text and metadata when `mirror_base` is not specified:
# ```python
# def test_create_doc_from_plain_html_content_without_mirror_base():
#     url = "http://example.com"
#     plain_html_content = "<html><body>Test content</body></html>"
#     expected_text = "Test content"
#     expected_metadata = {
#         "source_id": url,
#         "source_type": "html",
#         "simple_id": create_simple_identifier_from_url(url)
#     }
#     document = create_doc_from_plain_html_content(url, plain_html_content)
#     assert document.text == expected_text
#     assert document.metadata == expected_metadata
# ```



# METHOD:
# def get_documents_from_urls_as_mirror(mirror_url: str, producer_sink=lambda: Document) -> List[Document]:
# def test_no_urls():
#     assert get_documents_from_urls_as_mirror("") == []

# def test_valid_urls():
#     urls = ["http://example.com", "https://example.org"]
#     assert get_documents_from_urls_as_mirror("", urls) == [Document(url=urls[0]), Document(url=urls[1])]

# def test_invalid_url():
#     with pytest.raises(ValueError):
#         get_documents_from_urls_as_mirror("", ["not a url"])

# def test_connection_error():
#     with pytest.raises(ConnectionError):
#         get_documents_from_urls_as_mirror("", ["http://example.com"])

# def test_timeout_error():
#     with pytest.raises(TimeoutError):
#         get_documents_from_urls_as_mirror("", ["http://example.com"])


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_documents_from_urls_as_mirror` method:
# 
# 1. Test that the method returns an empty list if no URLs are provided:
# ```python
# def test_no_urls():
#     assert get_documents_from_urls_as_mirror("") == []
# ```
# 2. Test that the method returns a list of documents if valid URLs are provided:
# ```python
# def test_valid_urls():
#     urls = ["http://example.com", "https://example.org"]
#     assert get_documents_from_urls_as_mirror("", urls) == [Document(url=urls[0]), Document(url=urls[1])]
# ```
# 3. Test that the method raises a `ValueError` if an invalid URL is provided:
# ```python
# def test_invalid_url():
#     with pytest.raises(ValueError):
#         get_documents_from_urls_as_mirror("", ["not a url"])
# ```
# 4. Test that the method raises a `ConnectionError` if a URL cannot be reached:
# ```python
# def test_connection_error():
#     with pytest.raises(ConnectionError):
#         get_documents_from_urls_as_mirror("", ["http://example.com"])
# ```
# 5. Test that the method raises a `TimeoutError` if a URL takes too long to respond:
# ```python
# def test_timeout_error():
#     with pytest.raises(TimeoutError):
#         get_documents_from_urls_as_mirror("", ["http://example.com"])
# ```



# METHOD:
# def get_documents_from_urls_as_mirror_rec(mirror_base: str, current_url: str, already_seen_urls: List[str], producer_sink=lambda: Document) -> List[Document]:
# def test_no_urls():
#     assert get_documents_from_urls_as_mirror_rec("", [], []) == []

# def test_with_urls():
#     urls = ["http://example.com/page1", "http://example.com/page2"]
#     already_seen_urls = []
#     producer_sink = lambda: Document
#     result = get_documents_from_urls_as_mirror_rec("", urls, already_seen_urls, producer_sink)
#     assert len(result) == 2

# def test_skip_non_http_url():
#     urls = ["ftp://example.com/page1", "http://example.com/page2"]
#     already_seen_urls = []
#     producer_sink = lambda: Document
#     result = get_documents_from_urls_as_mirror_rec("", urls, already_seen_urls, producer_sink)
#     assert len(result) == 1

# def test_skip_already_seen_url():
#     urls = ["http://example.com/page1", "http://example.com/page2"]
#     already_seen_urls = ["http://example.com/page1"]
#     producer_sink = lambda: Document
#     result = get_documents_from_urls_as_mirror_rec("", urls, already_seen_urls, producer_sink)
#     assert len(result) == 1

# def test_process_unseen_url():
#     urls = ["http://example.com/page1", "http://example.com/page2"]
#     already_seen_urls = []
#     producer_sink = lambda: Document
#     result = get_documents_from_urls_as_mirror_rec("", urls, already_seen_urls, producer_sink)
#     assert len(result) == 2

# def test_process_unseen_url_with_content():
#     urls = ["http://example.com/page1", "http://example.com/page2"]
#     already_seen_urls = []
#     producer_sink = lambda: Document
#     result = get_documents_from_urls_as_mirror_rec("", urls, already_seen_urls, producer_sink)
#     assert len(result) == 2
#     for doc in result:
#         assert doc.content == "Some content"


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_documents_from_urls_as_mirror_rec` method:
# 
# 1. Test that the function returns an empty list when there are no URLs to process:
# ```python
# def test_no_urls():
#     assert get_documents_from_urls_as_mirror_rec("", [], []) == []
# ```
# 2. Test that the function returns a list of documents when there are URLs to process:
# ```python
# def test_with_urls():
#     urls = ["http://example.com/page1", "http://example.com/page2"]
#     already_seen_urls = []
#     producer_sink = lambda: Document
#     result = get_documents_from_urls_as_mirror_rec("", urls, already_seen_urls, producer_sink)
#     assert len(result) == 2
# ```
# 3. Test that the function skips URLs that are not starting with "http":
# ```python
# def test_skip_non_http_url():
#     urls = ["ftp://example.com/page1", "http://example.com/page2"]
#     already_seen_urls = []
#     producer_sink = lambda: Document
#     result = get_documents_from_urls_as_mirror_rec("", urls, already_seen_urls, producer_sink)
#     assert len(result) == 1
# ```
# 4. Test that the function skips URLs that are already seen:
# ```python
# def test_skip_already_seen_url():
#     urls = ["http://example.com/page1", "http://example.com/page2"]
#     already_seen_urls = ["http://example.com/page1"]
#     producer_sink = lambda: Document
#     result = get_documents_from_urls_as_mirror_rec("", urls, already_seen_urls, producer_sink)
#     assert len(result) == 1
# ```
# 5. Test that the function processes URLs that are not seen before and returns a list of documents:
# ```python
# def test_process_unseen_url():
#     urls = ["http://example.com/page1", "http://example.com/page2"]
#     already_seen_urls = []
#     producer_sink = lambda: Document
#     result = get_documents_from_urls_as_mirror_rec("", urls, already_seen_urls, producer_sink)
#     assert len(result) == 2
# ```
# 6. Test that the function processes URLs that are not seen before and returns a list of documents with the correct content:
# ```python
# def test_process_unseen_url_with_content():
#     urls = ["http://example.com/page1", "http://example.com/page2"]
#     already_seen_urls = []
#     producer_sink = lambda: Document
#     result = get_documents_from_urls_as_mirror_rec("", urls, already_seen_urls, producer_sink)
#     assert len(result) == 2
#     for doc in result:
#         assert doc.content == "Some content"
# ```
# Note that these tests are just examples and may need to be adjusted based on the specific requirements of your project.

