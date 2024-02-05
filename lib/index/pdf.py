
from typing import List
from llama_index import Document, download_loader

def get_content_from_pdf_file(pdf_file_path: str) -> List[Document]:
    # https://docs.llamaindex.ai/en/stable/api_reference/readers.html
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    return loader.load_data(file=pdf_file_path, extra_info={"source_id": pdf_file_path})
