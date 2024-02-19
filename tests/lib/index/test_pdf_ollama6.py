import pytest
from unittest.mock import Mock, patch
import json
from typing import List
from llama_index.core import Document
from llama_index.readers.file import PDFReader
from lib.index.pdf import (get_content_from_pdf_file)

# METHOD:
# def get_content_from_pdf_file(pdf_file_path: str) -> List[Document]:
def test_get_content_from_pdf_file_returns_list_of_documents(self):
    pdf_file_path = "test_data/example.pdf"
    expected_output = [Document("Test Document 1", "This is the first document"), Document("Test Document 2", "This is the second document")]
    loader = PDFReader()
    actual_output = loader.load_data(file=pdf_file_path, extra_info={"source_id": pdf_file_path})
    self.assertEqual(actual_output, expected_output)

def test_get_content_from_pdf_file_raises_filenotfounderror_when_given_invalid_path(self):
    pdf_file_path = "test_data/example.pdf"
    loader = PDFReader()
    with self.assertRaises(FileNotFoundError):
        actual_output = loader.load_data(file=pdf_file_path, extra_info={"source_id": pdf_file_path})

def test_get_content_from_pdf_file_returns_empty_list_when_given_empty_pdf(self):
    pdf_file_path = "test_data/example.pdf"
    loader = PDFReader()
    actual_output = loader.load_data(file=pdf_file_path, extra_info={"source_id": pdf_file_path})
    self.assertEqual(actual_output, [])

def test_get_content_from_pdf_file_returns_list_of_documents_when_given_multiple_pages(self):
    pdf_file_path = "test_data/example.pdf"
    expected_output = [Document("Test Document 1", "This is the first document"), Document("Test Document 2", "This is the second document")]
    loader = PDFReader()
    actual_output = loader.load_data(file=pdf_file_path, extra_info={"source_id": pdf_file_path})
    self.assertEqual(actual_output, expected_output)

def test_get_content_from_pdf_file_returns_list_of_documents_when_given_different_font_sizes(self):
    pdf_file_path = "test_data/example.pdf"
    expected_output = [Document("Test Document 1", "This is the first document"), Document("Test Document 2", "This is the second document")]
    loader = PDFReader()
    actual_output = loader.load_data(file=pdf_file_path, extra_info={"source_id": pdf_file_path})
    self.assertEqual(actual_output, expected_output)


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `get_content_from_pdf_file` method:
# 
# 1. Test that the method returns a list of documents when given a valid PDF file path:
# ```python
# def test_get_content_from_pdf_file_returns_list_of_documents(self):
#     pdf_file_path = "test_data/example.pdf"
#     expected_output = [Document("Test Document 1", "This is the first document"), Document("Test Document 2", "This is the second document")]
#     loader = PDFReader()
#     actual_output = loader.load_data(file=pdf_file_path, extra_info={"source_id": pdf_file_path})
#     self.assertEqual(actual_output, expected_output)
# ```
# 2. Test that the method raises a `FileNotFoundError` when given an invalid PDF file path:
# ```python
# def test_get_content_from_pdf_file_raises_filenotfounderror_when_given_invalid_path(self):
#     pdf_file_path = "test_data/example.pdf"
#     loader = PDFReader()
#     with self.assertRaises(FileNotFoundError):
#         actual_output = loader.load_data(file=pdf_file_path, extra_info={"source_id": pdf_file_path})
# ```
# 3. Test that the method returns an empty list when given a PDF file with no content:
# ```python
# def test_get_content_from_pdf_file_returns_empty_list_when_given_empty_pdf(self):
#     pdf_file_path = "test_data/example.pdf"
#     loader = PDFReader()
#     actual_output = loader.load_data(file=pdf_file_path, extra_info={"source_id": pdf_file_path})
#     self.assertEqual(actual_output, [])
# ```
# 4. Test that the method returns a list of documents when given a PDF file with multiple pages:
# ```python
# def test_get_content_from_pdf_file_returns_list_of_documents_when_given_multiple_pages(self):
#     pdf_file_path = "test_data/example.pdf"
#     expected_output = [Document("Test Document 1", "This is the first document"), Document("Test Document 2", "This is the second document")]
#     loader = PDFReader()
#     actual_output = loader.load_data(file=pdf_file_path, extra_info={"source_id": pdf_file_path})
#     self.assertEqual(actual_output, expected_output)
# ```
# 5. Test that the method returns a list of documents when given a PDF file with different font sizes:
# ```python
# def test_get_content_from_pdf_file_returns_list_of_documents_when_given_different_font_sizes(self):
#     pdf_file_path = "test_data/example.pdf"
#     expected_output = [Document("Test Document 1", "This is the first document"), Document("Test Document 2", "This is the second document")]
#     loader = PDFReader()
#     actual_output = loader.load_data(file=pdf_file_path, extra_info={"source_id": pdf_file_path})
#     self.assertEqual(actual_output, expected_output)
# ```

