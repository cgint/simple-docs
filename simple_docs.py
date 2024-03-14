import asyncio
import os
import time
from lib.hybrid_query import get_hybrid_query_engine

from lib.index.helper import cur_simple_date_time_sec
from lib.index.error_helper import write_error_to_file
from lib.llm import get_aim_callback, get_embed_model, get_llm
from lib.index.index import index
from llama_index.core import Settings
import llama_index
from lib import constants
from llama_index.core.query_engine import RetrieverQueryEngine

# TODOS
#  - "Some text is provided below. Given the text, extract up to 10 knowledge triplets in the form of"
#    - Max. 10 kg-triplets good value ?
#    - Most domain/semantically relevant first!
#  - Lots of e.g. "h2rpczKP4RKUjjMy0pUCSp/eSdY1CHlF7PI0eOzOQ95xg3UU4ZKMb3Ix2K3g/pIc8h0SXcQ8seac7GOeoX3wKENYJBI+R2T7CQAUCzxq/o/vB07k6o29RZ+kQkkrwQupxvJBofR8ME6bAYM28yJtw2dTqGembOXC7DBEbU12hszX25lnbj+OOP79ajGN0SHreMnyF/cUSIhHTVc3BP5stVonaFc/0xPqkqA8eI1gemzCl4Howw+H4Zq1DHEI7T9pJOc4N84pN/akO7M3iN11TtmyiuQsXHjIARMAJGwAgMQgRMFHfmS4tEsbyaVFKtGcNvDfg0GMTik4mp1kqpGvRFIksTk1bS4+GnCYbup7KxlWVtVJyLCMMjkkmtrEdJj2KbCRiDeP54Fgbysh7tbSgc7mHpHATqEcXUFyaTKEzOOeecPLHpST3RhCdau+vp49pVKDuYYCKt1H8m8o3uEb0PRXhyj0jYSvnDc"
#   - Should not be considered! Waste of resources!
#   - Most probably an result of trating an xml as plain text!
#  - Use json reader for json files!
#  - strip html better so that
#      how it compares to scuba diving and safe practice tips<br/></p></td></tr><tr><td class="confluenceTd"><p>Running Spark Applications on Kubernetes
#    does not become
#      how it compares to scuba diving and safe practice tipsRunning Spark Applications on Kubernetes
#    and we created a useless word 'tipsRunning'!
#   - lots of tuples instead of triplets! e.g. "(Click ID in Conversion Tag, checker checks whether click ID sent to Google)"
#     - using 'neural-chat' model. although also lots of triplets!
#     - Better prompt ?
#     + maybe other delimiter as some triplets contain commas themselves!

exec_id = "ExecID: " + cur_simple_date_time_sec()

def init_service_context(llm_options, callback_handler = None):
    if callback_handler is not None:
        llama_index.global_handler = callback_handler
    llm = get_llm(llm_options["engine"], llm_options["model"], llm_options["openai_model"])
    embed = get_embed_model(llm_options["embed_engine"], llm_options["embed_model"])
    Settings.llm = llm
    Settings.embed_model = embed
    Settings.chunk_size = 1024

def get_params_from_env():
    command = os.environ.get("PARAM_COMMAND", "")
    if command != "index" and command != "ask":
        print("Usage:")
        print("  PARAM_COMMAND: 'index', 'ask'")
        # print(f"  Data from {index_dir} will be indexed and moved to {index_dir_done} when done.")
        exit(1)
    fixed_questions = os.environ.get("ASK_SENTENCES")
    llm_options = {
        "engine": os.environ.get("LLM_ENGINE", "ollama"),
        "model": os.environ.get("LLM_MODEL", "neural-chat"),
        "embed_engine": os.environ.get("LLM_EMBED_ENGINE", "fastembed"),
        "embed_model": os.environ.get("LLM_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "openai_model": os.environ.get("OPENAI_MODEL")
    }
    query_engine_options = {
        "variant": os.environ.get("QUERY_ENGINE_VARIANT", "vector-graph-docsum-term"),
        "retriever_k": os.environ.get("RETRIEVER_K", "10"),
        "reranker_k": os.environ.get("RERANKER_K", "14000"),
        "reranker_model": os.environ.get("RERANKER_MODEL", "query-wordmatch-charsum") # "BAAI/bge-reranker-base")
    }
    indexing_engine_options = {
        "variant": os.environ.get("INDEXING_ENGINE_VARIANT", "vector-graph-docsum-term")
    }
    return command, fixed_questions, llm_options, query_engine_options, indexing_engine_options

def get_aim_callback_handler(exec_id, llm_options, query_engine_options, command, question_info = None):
    experiment_name = f"{llm_options['engine']}_{llm_options['model']}" + (f"_{llm_options['openai_model']}" if llm_options['openai_model'] is not None else "")
    aim_run_params = { 
        'exec_id': exec_id,
        'query_engine_options': query_engine_options, 
        'command': command,
        'llm': llm_options 
        }
    if question_info is not None:
        aim_run_params['question_info'] = question_info
    return get_aim_callback(experiment_name, constants.aim_dir, aim_run_params)

async def ask_question(query_engine: RetrieverQueryEngine, question: str):
    print(f"Finding an answer for question: {question}")
    start_time = int(round(time.time() * 1000))
    try:
        answer_full = await query_engine.aquery(question)
        # print(answer_full.source_nodes)
        answer = answer_full.response.strip()
    except Exception as e:
        msg = f"Error processing question '{question}' ... skipping and continuing with next: {e}"
        write_error_to_file(e, msg)
        answer = f"An error occurred: {e}"
    if answer == "" or answer == None:
        answer = "No answer found."
    end_time = int(round(time.time() * 1000))
    duration_sec = (end_time - start_time) / 1000
    print(f"Q: {question}\nA: {answer}")
    print("\nAnswering took: " + str(duration_sec) + " sec\n")
    # print(f"  Source Nodes ({len(answer_full.source_nodes)}) :")
    # for n in answer_full.source_nodes:
    #     first_text_chars = n.node.text[:30]
    #     print(f"    Source Node: {n.score} - {n.node_id} - {first_text_chars} - {answer_full.metadata[n.node_id]}")
    # print("\n")

def create_directories():
    os.makedirs(constants.embed_cache_dir, exist_ok=True)
    os.makedirs(constants.index_dir, exist_ok=True)
    os.makedirs(constants.index_dir_done, exist_ok=True)
    os.makedirs(constants.html_dl_cache_dir, exist_ok=True)
    os.makedirs(constants.aim_dir, exist_ok=True)
    os.makedirs(constants.term_data_dir, exist_ok=True)
    # os.makedirs(constants.vector_storage_dir, exist_ok=True)
    # os.makedirs(constants.doc_sum_index_dir, exist_ok=True)
    # os.makedirs(constants.kg_graph_index_dir, exist_ok=True)

async def main():
    create_directories()
    command, fixed_questions, llm_options, query_engine_options, indexing_engine_options = get_params_from_env()
    if command != "index":
        print(f"Setting callback handler for AIM as global handler ...")
        llama_index.global_handler = get_aim_callback_handler(exec_id, llm_options, query_engine_options, command, fixed_questions)
    init_service_context(llm_options)
    print(f"Service Context created ...")
    if command != "index":
        print(f"Creating a query engine according to {query_engine_options} ...")
        query_engine_options['doc_sum_index_dir'] = constants.doc_sum_index_dir
        query_engine_options['collection'] = constants.collection
        query_engine_options['graph_db'] = constants.graph_db
        query_engine_options['kg_graph_index_dir'] = constants.kg_graph_index_dir
        query_engine = get_hybrid_query_engine(query_engine_options)
        print("Query engine created ...")

    if command == "index":
        indexing_engine_options['doc_sum_index_dir'] = constants.doc_sum_index_dir
        indexing_engine_options['collection'] = constants.collection
        indexing_engine_options['graph_db'] = constants.graph_db
        indexing_engine_options['kg_graph_index_dir'] = constants.kg_graph_index_dir
        index(indexing_engine_options, constants.index_dir, constants.index_dir_done)
    elif fixed_questions is not None and fixed_questions != "":
        fixed_questions = fixed_questions.split("###")
        for question in fixed_questions:
            await ask_question(query_engine, question)
    else:
        while True:
            question = input("Ask (type 'e' or 'end' to finish): ").strip()
            if question == "":
                continue
            if question == "end" or question == "e":
                break
            await ask_question(query_engine, question)
        print("\nBye.")


if __name__ == "__main__":
    asyncio.run(main())
