import os

from lib.index.helper import cur_simple_date_time_sec
from lib.llm import get_aim_callback, get_embed_model, get_llm
from lib.index.index import index
from llama_index import ServiceContext
from lib.ask import ask_hybrid_query_engine
import llama_index

index_dir = "/data/index_inbox"
index_dir_done = "/data/index_inbox/done"

vector_storage_dir = "/data/vector_index"
collection = vector_storage_dir.replace("/", "_").replace("_", "")
doc_sum_index_dir = "/data/doc_sum_index"
aim_dir = "/data/aim"
g_db = "neo4j"

exec_id = "ExecID: " + cur_simple_date_time_sec()

def get_service_context(llm_options, callback_handler = None):
    if callback_handler is not None:
        import llama_index
        llama_index.global_handler = callback_handler
    llm = get_llm(llm_options["engine"], llm_options["model"], llm_options["openai_model"])
    embed = get_embed_model(llm_options["embed_engine"], llm_options["embed_model"])
    sc = ServiceContext.from_defaults(
        llm=llm, 
        chunk_size=512,
        embed_model=embed
    )
    return sc

def get_params_from_env():
    command = os.environ.get("PARAM_COMMAND", "")
    if command != "index" and command != "ask":
        print("Usage:")
        print("  PARAM_COMMAND: 'index', 'ask'")
        print(f"  Data from {index_dir} will be indexed and moved to {index_dir_done} when done.")
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
    return command, fixed_questions, llm_options, query_engine_options

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
    return get_aim_callback(experiment_name, aim_dir, aim_run_params)

def ask_question(service_context, query_engine_options, doc_sum_index_dir, question):
    print(f"Looking for an answer for question: {question}")
    answer_full = ask_hybrid_query_engine(service_context, query_engine_options, doc_sum_index_dir, collection, g_db, question)
    answer = answer_full.response
    if answer == "" or answer == None:
        answer = "No answer found."
    print("Answer: " + answer)
    print("\n")
    # print(f"  Source Nodes ({len(answer_full.source_nodes)}) :")
    # for n in answer_full.source_nodes:
    #     first_text_chars = n.node.text[:30]
    #     print(f"    Source Node: {n.score} - {n.node_id} - {first_text_chars} - {answer_full.metadata[n.node_id]}")
    # print("\n")

if __name__ == "__main__":
    command, fixed_questions, llm_options, query_engine_options = get_params_from_env()
    llama_index.global_handler = get_aim_callback_handler(exec_id, llm_options, query_engine_options, command, fixed_questions)
    service_context = get_service_context(llm_options)

    if fixed_questions is not None and fixed_questions != "":
        fixed_questions = fixed_questions.split("###")
        for question in fixed_questions:
            ask_question(service_context, query_engine_options, doc_sum_index_dir, question)
    elif command == "index":
        index(service_context, doc_sum_index_dir, collection, g_db, index_dir, index_dir_done)
    else:
        while True:
            question = input("Ask (type 'e' or 'end' to finish): ").strip()
            if question == "":
                continue
            if question == "end" or question == "e":
                break
            ask_question(service_context, query_engine_options, doc_sum_index_dir, question)
        print("\nBye.")