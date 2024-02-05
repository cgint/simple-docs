import os
from llama_index import DocumentSummaryIndex, StorageContext, get_response_synthesizer, load_index_from_storage
from llama_index.query_engine import BaseQueryEngine

def create_doc_summary_index(documents_remembered, service_context, storage_dir):
    if os.path.exists(storage_dir):
        idx = load_index_from_storage(StorageContext.from_defaults(persist_dir=storage_dir))
        print(f"Refreshing {len(documents_remembered)} documents in existing doc_sum_index.")
        idx.refresh_ref_docs(documents_remembered)
    else:
        print(f"Refreshing {len(documents_remembered)} documents in newly to be created doc_sum_index.")
        idx = DocumentSummaryIndex.from_documents(
            documents_remembered,
            service_context=service_context,
            show_progress=True,
        )
    idx.storage_context.persist(persist_dir=storage_dir)

def load_doc_sum_index(storage_dir):
    return load_index_from_storage(StorageContext.from_defaults(persist_dir=storage_dir))

def get_doc_sum_index_query_engine(service_context, storage_dir: str) -> BaseQueryEngine:
    return load_doc_sum_index(storage_dir).as_query_engine(
        service_context=service_context,
        response_synthesizer=get_response_synthesizer(
            service_context=service_context,
            response_mode="tree_summarize", 
            use_async=True
        )
    )
