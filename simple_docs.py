import asyncio
import os
from lib.hybrid_query import get_hybrid_query_engine

from lib.index.helper import cur_simple_date_time_sec
from lib.llm import get_aim_callback, get_embed_model, get_llm
from lib.index.index import index
from llama_index import ServiceContext
import llama_index
from lib import constants
from llama_index.query_engine import RetrieverQueryEngine

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
    try:
        answer_full = await query_engine.aquery(question)
        # print(answer_full.source_nodes)
        answer = answer_full.response.strip()
    except Exception as e:
        answer = f"An error occurred: {e}"
        print(answer)
    if answer == "" or answer == None:
        answer = "No answer found."
    print(f"Q: {question}\nA: {answer}")
    print("\n")
    # print(f"  Source Nodes ({len(answer_full.source_nodes)}) :")
    # for n in answer_full.source_nodes:
    #     first_text_chars = n.node.text[:30]
    #     print(f"    Source Node: {n.score} - {n.node_id} - {first_text_chars} - {answer_full.metadata[n.node_id]}")
    # print("\n")


async def main():
    command, fixed_questions, llm_options, query_engine_options, indexing_engine_options = get_params_from_env()
    if command != "index":
        print(f"Setting callback handler for AIM as global handler ...")
        llama_index.global_handler = get_aim_callback_handler(exec_id, llm_options, query_engine_options, command, fixed_questions)
    service_context = get_service_context(llm_options)
    print(f"Service Context created ...")
    if command != "index":
        print(f"Creating a query engine according to {query_engine_options} ...")
        query_engine_options['doc_sum_index_dir'] = constants.doc_sum_index_dir
        query_engine_options['collection'] = constants.collection
        query_engine_options['graph_db'] = constants.graph_db
        query_engine_options['kg_graph_index_dir'] = constants.kg_graph_index_dir
        query_engine = get_hybrid_query_engine(service_context, query_engine_options)
        print("Query engine created ...")

    if command == "index":
        indexing_engine_options['doc_sum_index_dir'] = constants.doc_sum_index_dir
        indexing_engine_options['collection'] = constants.collection
        indexing_engine_options['graph_db'] = constants.graph_db
        indexing_engine_options['kg_graph_index_dir'] = constants.kg_graph_index_dir
        index(service_context, indexing_engine_options, constants.index_dir, constants.index_dir_done)
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
