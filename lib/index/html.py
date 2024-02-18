
from typing import List
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from llama_index.core import Document

from lib.index.web import create_simple_identifier_from_url, get_plain_content_from

def get_documents_from_urls(urls: List[str], producer_sink=lambda: Document) -> List[Document]:
    for url in urls:
        if url.startswith("http"):
            print(f"Indexing {url} ...")
            producer_sink(create_doc_from_plain_html_content(url, get_plain_content_from(url)))  
        else:
            print(f"Skipping {url} ...")

def extract_content_part_from_html(plain_html_content: str, initial_tag: str = "body") -> BeautifulSoup:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(plain_html_content, 'html.parser')
    # fetch body only
    content_part = soup.find(initial_tag) if soup.find(initial_tag) is not None else None
    print(f"1) {content_part}")
    if content_part is None:
        print(f"1a) {content_part}")
        content_part = soup.find('body') if soup.find('body') is not None else None
    print(f"2) {content_part}")
    if content_part is None:
        return soup
    print(f"3) {content_part}")
    # remove script and style tags
    for tagsToRemove in content_part(["script", "style"]):
        tagsToRemove.extract()
    print(f"4) {content_part}")
    return content_part

def clean_html_content(plain_html_content: str) -> str:
    return extract_content_part_from_html(plain_html_content, "article").get_text()

def get_urls_from_html_content(plain_html_content: str) -> List[str]:
    content_part = extract_content_part_from_html(plain_html_content)
    return [a["href"] for a in content_part.find_all("a", href=True)]

def create_doc_from_plain_html_content(url, plain_html_content: str, mirror_base=None) -> Document:
    meta = {"source_id": url, "source_type": "html", "simple_id": create_simple_identifier_from_url(url)}
    if mirror_base is not None:
        meta["mirror_base"] = mirror_base
    return Document(text=clean_html_content(plain_html_content), metadata=meta)

def get_documents_from_urls_as_mirror(mirror_url: str, producer_sink=lambda: Document) -> List[Document]:
    return get_documents_from_urls_as_mirror_rec(mirror_url, mirror_url, [], producer_sink)

def get_documents_from_urls_as_mirror_rec(mirror_base: str, current_url: str, already_seen_urls: List[str], producer_sink=lambda: Document) -> List[Document]:
    if not current_url.startswith("http"):
        # print(f"Not starting with 'http' {current_url} ...")
        return
    if current_url in already_seen_urls:
        # print(f"Skipping {current_url} as already seen ...")
        return
    # print(f"HTML Indexing {current_url} ...")
    already_seen_urls.append(current_url)
    content = get_plain_content_from(current_url)
    producer_sink(create_doc_from_plain_html_content(current_url, content, mirror_base))  
    contained_urls = get_urls_from_html_content(content)
    urls_full_path_no_hash = [urljoin(current_url, potential_sub_url.split("#")[0]) for potential_sub_url in contained_urls]
    urls_is_sub_page = [potential_sub_url for potential_sub_url in urls_full_path_no_hash if potential_sub_url.startswith(mirror_base)]
    sub_urls_unseen = [potential_sub_url for potential_sub_url in urls_is_sub_page if potential_sub_url not in already_seen_urls]
    # print(f"  -> Found {len(contained_urls)} urls, {len(urls_is_sub_page)} sub pages and {len(sub_urls_unseen)} unseen in {current_url}.")
    for sub_url in sub_urls_unseen:
        get_documents_from_urls_as_mirror_rec(mirror_base, sub_url, already_seen_urls, producer_sink)

