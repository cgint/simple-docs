from llama_index_client import Document
import pytest
from lib.index.html import (extract_content_part_from_html,
                              clean_html_content, get_urls_from_html_content,
                              create_doc_from_plain_html_content)
from unittest.mock import Mock, patch
import lib.index.html  # Assuming this is the module where get_documents_from_urls is defined


@pytest.mark.parametrize("input_html, content_part", [
    ("<html><head></head><body><p>Test content</p></body></html>", "<body><p>Test content</p></body>"),
    ("<html><body><div>Another test</div></body></html>", "<body><div>Another test</div></body>"),
    
    ("<div>Only part of html</div>", "<div>Only part of html</div>"),
    ("Not actually html", "Not actually html"),
])
def test_extract_content_part_from_html(input_html, content_part):
    assert str(extract_content_part_from_html(input_html)) == content_part

@pytest.mark.parametrize("html_content, clean_content", [
    ("<html><body><article>Test content</article></body></html>", "Test content"),
    ("<html><body><table><tr><td>Cell11  </td><td>Cell12</td></tr><tr><td> Cell21</td><td>Cell22</td></tr></table></article></body></html>",
     "Cell11 Cell12 Cell21 Cell22"),
])
def test_clean_html_content(html_content, clean_content):
    assert clean_html_content(html_content) == clean_content

def test_get_urls_from_html_content():
    html_content = '<html><body><a href="http://example.com">Example</a></body></html>'
    result = get_urls_from_html_content(html_content)
    assert "http://example.com" in result

def test_create_doc_from_plain_html_content():
    url = "http://example.com"
    html_content = "<html><body><article>Test content</article></body></html>"
    result = create_doc_from_plain_html_content(url, html_content)
    assert result.text.strip() == "Test content"
    assert result.metadata['source_id'] == url
    assert result.metadata['source_type'] == "html"

def test_get_documents_from_urls_with_mock():
    with patch('lib.index.html.get_plain_content_from') as mock_get_content:
        mock_get_content.return_value = "Dummy response"
        mock_producer_sink = Mock()
        lib.index.html.get_documents_from_urls(["http://test.com/"], mock_producer_sink)
        mock_get_content.assert_called_with("http://test.com/")
        mock_producer_sink.assert_called_once()
        doc: Document = mock_producer_sink.call_args[0][0]
        assert doc.text == "Dummy response"
        assert doc.extra_info == {"source_id": "http://test.com/", "source_type": "html", "simple_id": "http___test_com_"}
        
def test_get_documents_from_urls_with_invalid_urls():
    with patch("lib.index.web.get_plain_content_from_url_or_cache") as mock_get_content:
        mock_producer_sink = Mock()
        lib.index.html.get_documents_from_urls(["hp://test.com"], mock_producer_sink)
        mock_get_content.assert_not_called()
        mock_producer_sink.assert_not_called()

# def test_get_documents_from_urls_with_valid_urls():
#     mock_producer_sink = Mock()
#     urls = ["http://example.com", "http://test.com"]
#     get_documents_from_urls(urls, mock_producer_sink)
#     assert mock_producer_sink.call_count == 2

# def test_get_documents_from_urls_with_invalid_urls():
#     mock_producer_sink = Mock()
#     urls = ["invalid_url", "http://test.com"]
#     get_documents_from_urls(urls, mock_producer_sink)
#     assert mock_producer_sink.call_count == 1