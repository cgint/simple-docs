from typing import List
from llama_index.core import Document
from llama_index.core.readers.json import JSONReader

def get_content_from_json_file(json_file_path: str, source_type: str, is_jsonl: bool = False) -> List[Document]:
    # https://docs.llamaindex.ai/en/stable/api_reference/readers.html
    loader = JSONReader(is_jsonl=is_jsonl, levels_back=0, collapse_length=1000)
    docs = loader.load_data(json_file_path)
    for doc in docs:
        doc.metadata["source_id"] = json_file_path
        doc.metadata["source_type"] = source_type
    return docs
#, extra_info={"source_id": json_file_path})
