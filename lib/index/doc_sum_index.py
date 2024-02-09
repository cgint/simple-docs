import os
from llama_index import DocumentSummaryIndex, StorageContext, get_response_synthesizer, load_index_from_storage
from llama_index.query_engine import BaseQueryEngine

def create_doc_summary_index(documents, service_context, storage_dir):
    return operate_on_doc_sum_index(
        service_context,
        storage_dir,
        lambda doc_sum_index: doc_sum_index.refresh_ref_docs(documents)
    )

def operate_on_doc_sum_index(service_context, doc_sum_index_dir: str, operation=lambda: None) -> DocumentSummaryIndex:
    idx = get_doc_sum_index(service_context, doc_sum_index_dir)
    try:
        operation(idx)
    finally:
        doc_count = len(idx.docstore.docs)
        print(f"Persisting {doc_count} docs for doc_sum_index to {doc_sum_index_dir} ...")
        idx.storage_context.persist(persist_dir=doc_sum_index_dir)
    return idx

def get_doc_sum_index(service_context, doc_sum_index_dir):
    return load_doc_sum_index(service_context, doc_sum_index_dir)

def load_doc_sum_index(service_context, doc_sum_index_dir):
    if not os.path.exists(doc_sum_index_dir):
        print(f"Creating doc_sum_index in {doc_sum_index_dir} ...")
        idx = DocumentSummaryIndex.from_documents(
            [],
            service_context=service_context,
            show_progress=True,
        )
        idx.storage_context.persist(persist_dir=doc_sum_index_dir)
    sc = StorageContext.from_defaults(persist_dir=doc_sum_index_dir)
    return load_index_from_storage(sc, service_context=service_context, show_progress=True)

def get_doc_sum_index_query_engine(service_context, doc_sum_index_dir: str) -> BaseQueryEngine:
    return load_doc_sum_index(service_context, doc_sum_index_dir).as_query_engine(
        service_context=service_context,
        response_synthesizer=get_response_synthesizer(
            service_context=service_context,
            response_mode="tree_summarize", 
            use_async=True
        )
    )
