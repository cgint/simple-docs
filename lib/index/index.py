from concurrent.futures import FIRST_COMPLETED, ALL_COMPLETED, ThreadPoolExecutor, wait
import os
from time import sleep
from lib.index.doc_sum_index import delete_doc_summary_index, operate_on_doc_sum_index, persist_index

from lib.index.html import clean_html_content, get_documents_from_urls_as_mirror, get_documents_from_urls
from lib.index.helper import cur_simple_date_time_sec
from lib.index.kg_classic import delete_kg_graph_index, load_kg_graph_index, operate_on_kg_graph_index
from lib.index.pdf import get_content_from_pdf_file
from lib.index.terms.term_index import build_term_reference_index, count_terms_per_document, write_term_references_to_file
from lib.index.terms.terms import terms_from_txt
from lib.index.terms.kg_num_term_neo4j import kg_neo4j_delete_all_nodes, operate_on_graph_index
from lib.index.helper import list_files
from lib.json import get_doc_from_json
from lib.vector_chroma import delete_chroma_collection, operate_on_vector_index
from llama_index.core import Document, KnowledgeGraphIndex
from typing import List
from lib import constants
import queue
import threading

main_queue_size = 1
worker_queue_sizes = 5
consume_documents_threading_workers = 4
RETRY_ATTEMPTS_MAX = 3
DOC_SUM_PERSIST_EVERY = 50

def init_consumer_threads(indexing_engine_options):
    queues_map = {
        "vector": {
            "queue": queue.Queue(maxsize=worker_queue_sizes),
            "target": index_consume_documents_on_vector,
            "baseArgs": (indexing_engine_options['collection'], "VectorIndex")
        },
        "graph": {
            "queue": queue.Queue(maxsize=worker_queue_sizes),
            "target": index_consume_documents_on_graph,
            "baseArgs": (indexing_engine_options['graph_db'], "GraphIndex")
        },
        "kggraph": {
            "queue": queue.Queue(maxsize=worker_queue_sizes),
            "target": index_consume_documents_on_kg_graph,
            "baseArgs": (indexing_engine_options['kg_graph_index_dir'], "KGGraphIndex")
        },
        "term": {
            "queue": queue.Queue(maxsize=worker_queue_sizes),
            "target": index_consume_documents_on_term_index,
            "baseArgs": ("TermIndex",)
        },
        "docsum": {
            "queue": queue.Queue(maxsize=worker_queue_sizes),
            "target": index_consume_documents_on_doc_sum,
            "baseArgs": (indexing_engine_options['doc_sum_index_dir'], "DocSumIndex")
        }
    }
    # load_kg_graph_index(indexing_engine_options["kg_graph_index_dir"])

    fan_out_queues = []
    variant = indexing_engine_options["variant"] # e.g. vector-graph-term
    consumer_threads = []
    for k, v in queues_map.items():
        if k not in variant:
            continue
        fan_out_queues.append(v["queue"])
        args = v["baseArgs"] + (v["queue"],)
        consumer_threads.append(threading.Thread(target=v["target"], args=args))

    # Start consuming in parallel
    q_main = queue.Queue(maxsize=main_queue_size)
    consumer_threads.append(threading.Thread(target=queue_fan_out, args=(q_main, fan_out_queues)))
    for t in consumer_threads:
        t.start()
    
    return q_main, consumer_threads

def register_main_queue_push(q_main: queue.Queue, doc: Document):
    # doc_id = doc.doc_id if doc else "None"
    # q_size = q_main.qsize()
    # print(f"Pushing document '{doc_id}' to main_queue (size={q_size}) ...")
    q_main.put(doc)

def async_index(indexing_engine_options, index_dir, index_dir_done):
    if constants.del_indices_all:
        answer = input("Are you sure you want to delete all indices? (yes/no) ")
        if answer == "yes":
            print("\nDeleting on YOUR behalf ...\n")
            print("Deleting neo4j nodes ...")
            kg_neo4j_delete_all_nodes()
            print("Deleting chroma entries ...")
            delete_chroma_collection(indexing_engine_options["collection"])
            print("Deleting doc_sum index ...")
            delete_doc_summary_index(indexing_engine_options["doc_sum_index_dir"])
            print("Deleting kg_graph index ...")
            delete_kg_graph_index(indexing_engine_options["kg_graph_index_dir"])
    
    # Start producing
    try:
        q_main, consumer_threads = init_consumer_threads(indexing_engine_options)
        index_produce_documents(constants.max_files_to_index_per_run, index_dir, index_dir_done, lambda doc: register_main_queue_push(q_main, doc))
    finally:
        # Tell consumer to stop and wait for it to finish
        print("Telling consumers to finish once their queues are empty ...")
        q_main.put(None)
        for t in consumer_threads:
            t.join()

    # count docs per term
    write_term_references_to_file()
    count_terms_per_document()

def queue_fan_out(q, consumers):
    while True:
        doc = q.get()
        for c in consumers:
            c.put(doc)
        q.task_done()
        if doc is None:
            break  # Exit if None is received

def try_refresh_ref_docs_retry_max(attempts, idx, doc):
    for i in range(attempts):
        try:
            idx.refresh_ref_docs([doc])
            break
        except Exception as e:
            if i < 2:
                print(f" ERROR-RETRYING: Error refreshing ref_docs after a bit. The error: {e}")
                sleep(0.25)
            else:
                raise e

from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading

def index_consume_documents_threading_workers(log_name, q, persist=lambda: None, process=lambda x: None, max_threads=1, max_queued_tasks=5):
    counter = 0
    futures = set()

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        while True:
            counter += 1

            doc = q.get()
            if doc is None:
                q.task_done()
                break  # Exit if None is received

            # Check if we need to wait for some tasks to complete
            unfinished_count = len({future for future in futures if not future.done()})
            if unfinished_count >= max_queued_tasks:
                print(f" {log_name} - q(size={q.qsize()}) - Waiting for one of {unfinished_count} active tasks to complete before submitting next (backpressure) ...")
                # Wait for at least one task to complete
                _, _ = wait(futures, return_when=FIRST_COMPLETED)

            if counter % DOC_SUM_PERSIST_EVERY == 0:
                print(f" {log_name} - q(size={q.qsize()}) - Waiting for all running threads to finish before persisting ...")
                _, _ = wait(futures, return_when=ALL_COMPLETED)
                persist()

            # Submit new task
            futures.add(executor.submit(process, doc))
            print(f" {log_name} - q(size={q.qsize()}) - Indexing document #{counter} with id {doc.doc_id} ...")
            q.task_done()

        # Wait for all submitted tasks to complete
        for future in as_completed(futures):
            future.result()  # This also helps in raising exceptions if any occurred

def index_consume_documents_on_vector(collection, log_name, q):
    processor = lambda idx: index_consume_documents(log_name, q, lambda doc: try_refresh_ref_docs_retry_max(RETRY_ATTEMPTS_MAX, idx, doc))
    operate_on_vector_index(collection, processor)

def index_consume_documents_on_doc_sum(storage_dir, log_name, q):
    processor = lambda idx: index_consume_documents_threading_workers(log_name, q, 
        lambda: persist_index(idx, storage_dir), 
        lambda doc: try_refresh_ref_docs_retry_max(RETRY_ATTEMPTS_MAX, idx, doc), 
        consume_documents_threading_workers)
    operate_on_doc_sum_index(storage_dir, processor)

def index_consume_documents_on_graph(collection, log_name, q):
    processor = lambda idx: index_consume_documents(log_name, q, lambda doc: idx_update_doc_on_graph(idx, doc))
    operate_on_graph_index(collection, processor)

def index_consume_documents_on_kg_graph(kg_graph_index_dir, log_name, q):
    # processor = lambda idx: index_consume_documents(log_name, q, lambda doc: try_refresh_ref_docs_retry_max(RETRY_ATTEMPTS_MAX, idx, doc))
    processor = lambda idx: index_consume_documents_threading_workers(log_name, q, 
        lambda: persist_index(idx, kg_graph_index_dir), 
        lambda doc: try_refresh_ref_docs_retry_max(RETRY_ATTEMPTS_MAX, idx, doc), 
        consume_documents_threading_workers)
    operate_on_kg_graph_index(kg_graph_index_dir, processor)
    # operate_on_doc_sum_index(storage_dir, processor)

def index_consume_documents_on_term_index(log_name, q):
    index_consume_documents(log_name, q, build_term_reference_index)

def idx_update_doc_on_graph(idx: KnowledgeGraphIndex, doc: Document):
    terms = terms_from_txt(doc.text)
    for term in terms:
        idx.upsert_triplet_and_node((term, "in", doc.metadata["source_id"]), doc)

def index_consume_documents(log_name, q, process=lambda: None):
    counter = 0
    while True:
        doc = q.get()
        counter += 1
        if doc is None:
            q.task_done()
            break  # Exit if None is received
        print(f" {log_name} - q(size={q.qsize()}) - Indexing document #{counter} with id {doc.doc_id} ...")
        process(doc)
        q.task_done()

def index(indexing_engine_options, index_dir, index_dir_done):
    async_index(indexing_engine_options, index_dir, index_dir_done)


extensions_to_treat_as_plain_txt = ["tf", "yml", "yaml", "txt", "md", "java", "scala", "js", "ts", "xml", "kts", "gradle", "groovy", "py"]
known_extensions = ["json", "ndjson", "jsonl", "html", "pdf"] + extensions_to_treat_as_plain_txt
def index_produce_documents(max_files_to_process: int, index_dir: str, index_dir_done: str, producer_sink=lambda: Document):
    print(f"Looking for documents under {index_dir} to be indexed recursively ...")
    files_to_index = list_files(index_dir)
    if len(files_to_index) == 0:
        print(f"No documents to index in {index_dir}.")
        return
    run_index_time = cur_simple_date_time_sec()
    full_file_count = len(files_to_index)
    file_counter = 0
    print(f"Creating documents: {full_file_count} for index {run_index_time}")
    for file in files_to_index:
        file_full_path = file
        file_extension = file_full_path.split('.')[-1]
        if file_extension not in known_extensions:
            print(f"Skipping unknown file extension: {file_extension} for file {file_full_path}")
            continue
        file_counter += 1
        if max_files_to_process is not None and file_counter > max_files_to_process:
            print(f"Max files to process ({max_files_to_process}) reached. Will stop producing tasks ...")
            break
        print(f"Indexing {file_counter}/{full_file_count} {file} ...")
        try:
            process_file(file, file_full_path, file_extension, index_dir, index_dir_done, producer_sink)
        except Exception as e:
            print(f"Error processing {file_full_path} ... skipping and continuing with next: {e}")
            # write error including trace to file
            from traceback import format_exception
            with open(f"{index_dir_done}/error_{run_index_time}.log", "a") as f:
                f.write("========================================================================\n")
                f.write(f"================= {cur_simple_date_time_sec()} ==================\n")
                f.write("========================================================================\n")
                f.write(f"Error processing {file_full_path} ... skipping and continuing with next: {e}\n")
                f.write("Trace: "+"".join(format_exception(e))+"\n")
                f.write("========================================================================\n")
    

def process_file(file: str, file_full_path: str, file_extension: str, index_dir: str, index_dir_done: str, producer_sink=lambda: Document):
        single_file_processed = False
        if file == "fetch_urls.txt":
            get_documents_from_urls(read_relevant_lines(file_full_path), producer_sink)
        elif file == "mirror.txt":
            for url in read_relevant_lines(file_full_path):
                get_documents_from_urls_as_mirror(url, producer_sink)
        elif file_extension == "json":
            single_file_processed = True
            with open(file_full_path, "r") as f:
                metadata = {"source_id": file, "source_type": file_extension}
                producer_sink(get_doc_from_json(f.read(), metadata))
        elif file_extension == "jsonl" or file_extension == "ndjson":
            single_file_processed = True
            with open(file_full_path, "r") as f:
                for linenum, content in enumerate(f.read().splitlines()):
                    if content.strip() != "":
                        metadata = {"source_id": file, "source_type": "jsonl", "linenum": linenum}
                        producer_sink(get_doc_from_json(content, metadata))
        elif file_extension == "html":
            single_file_processed = True
            with open(file_full_path, "r") as f:
                producer_sink(Document(text=clean_html_content(f.read()), metadata={"source_id": file, "source_type": file_extension}))
        elif file_extension == "pdf":
            single_file_processed = True
            for doc in get_content_from_pdf_file(file_full_path):
                producer_sink(doc)
        elif file_extension in extensions_to_treat_as_plain_txt:
            single_file_processed = True
            with open(file_full_path, "r") as f:
                producer_sink(Document(text=f.read(), metadata={"source_id": file, "source_type": file_extension}))
        if single_file_processed:
            target_file_name = file.replace(index_dir, index_dir_done)
            # print(f" !!! NOT !!! - Moving {file_full_path} to {to_path} ... to preserve for the next run.")
            print(f"Moving {file_full_path} to {target_file_name} ...")
            os.makedirs(os.path.dirname(target_file_name), exist_ok=True)
            os.rename(file_full_path, target_file_name)
    

def read_relevant_lines(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f.read().splitlines() if line.strip() != "" and not line.strip().startswith("#")]
