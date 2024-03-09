import os
from typing import Set
from llama_index.core import DocumentSummaryIndex, StorageContext, get_response_synthesizer, load_index_from_storage
from llama_index.core.query_engine import BaseQueryEngine

def create_doc_summary_index(documents, storage_dir):
    return operate_on_doc_sum_index(
        storage_dir,
        lambda doc_sum_index: doc_sum_index.refresh_ref_docs(documents)
    )

def delete_doc_summary_index(doc_sum_index_dir: str):
    print(f"Deleting doc_sum_index at {doc_sum_index_dir} ...")
    if os.path.exists(doc_sum_index_dir):
        import shutil
        shutil.rmtree(doc_sum_index_dir)

def persist_index(idx: DocumentSummaryIndex, doc_sum_index_dir: str):
    doc_count = len(idx.docstore.docs)
    print(f"Persisting {doc_count} docs to {doc_sum_index_dir} ...")
    idx.storage_context.persist(persist_dir=doc_sum_index_dir)

def operate_on_doc_sum_index(doc_sum_index_dir: str, operation=lambda: None) -> DocumentSummaryIndex:
    import atexit
    idx = get_doc_sum_index(doc_sum_index_dir)
    atexist_reg_callable = atexit.register(persist_index, idx, doc_sum_index_dir)
    try:
        operation(idx)
    finally:
        persist_index(idx, doc_sum_index_dir)
        atexit.unregister(atexist_reg_callable)
    return idx

def get_doc_sum_index_doc_ids(doc_sum_index_dir: str, extract_key_from_doc=lambda: str) -> Set[str]:
    s = set()
    for doc in load_doc_sum_index(doc_sum_index_dir).docstore.docs.values():
        s.add(extract_key_from_doc(doc))
    return s

def get_doc_sum_index(doc_sum_index_dir) -> DocumentSummaryIndex:
    return load_doc_sum_index(doc_sum_index_dir)

def load_doc_sum_index(doc_sum_index_dir) -> DocumentSummaryIndex:
    if not os.path.exists(doc_sum_index_dir):
        print(f"Creating doc_sum_index in {doc_sum_index_dir} ...")
        idx = DocumentSummaryIndex.from_documents(
            [],
            show_progress=True,
        )
        persist_index(idx, doc_sum_index_dir)
    sc = StorageContext.from_defaults(persist_dir=doc_sum_index_dir)
    return load_index_from_storage(sc, show_progress=True)

def get_doc_sum_index_query_engine(doc_sum_index_dir: str) -> BaseQueryEngine:
    return load_doc_sum_index(doc_sum_index_dir).as_query_engine(
        response_synthesizer=get_response_synthesizer(
            response_mode="tree_summarize", 
            use_async=True
        )
    )
