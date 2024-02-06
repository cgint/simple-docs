import os
from llama_index import DocumentSummaryIndex, StorageContext, get_response_synthesizer, load_index_from_storage
from llama_index.query_engine import BaseQueryEngine

def create_doc_summary_index(documents, service_context, storage_dir):
    return operate_on_doc_sum_index(
        service_context,
        storage_dir,
        lambda doc_sum_index: doc_sum_index.refresh_ref_docs(documents)
    )

def operate_on_doc_sum_index(service_context, storage_dir: str, operation=lambda: None) -> DocumentSummaryIndex:
    idx = get_doc_sum_index(service_context, storage_dir)
    try:
        operation(idx)
    finally:
        idx.storage_context.persist(persist_dir=storage_dir)
    return idx

def get_doc_sum_index(service_context, storage_dir):
    return load_doc_sum_index(service_context, storage_dir)

def load_doc_sum_index(service_context, storage_dir):
    if not os.path.exists(storage_dir):
        print(f"Creating doc_sum_index in {storage_dir} ...")
        idx = DocumentSummaryIndex.from_documents(
            [],
            service_context=service_context,
            show_progress=True,
        )
        idx.storage_context.persist(persist_dir=storage_dir)
    return load_index_from_storage(StorageContext.from_defaults(persist_dir=storage_dir), show_progress=True)

def get_doc_sum_index_query_engine(service_context, storage_dir: str) -> BaseQueryEngine:
    return load_doc_sum_index(service_context, storage_dir).as_query_engine(
        service_context=service_context,
        response_synthesizer=get_response_synthesizer(
            service_context=service_context,
            response_mode="tree_summarize", 
            use_async=True
        )
    )
