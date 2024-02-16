from llama_index import Document

def get_doc_from_json(content: str, metadata: dict) -> Document:
    """
    I guess that is hackish ;) 
    Yet it is a way to create a Document object from a JSON string fitting this system.
    """
    return Document(text=content, metadata=metadata)

