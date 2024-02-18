import ast
from typing import List
from lib.index import helper
import os
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse
    )
sys_msg = """
You are a software quality senior engineer.
You write thorough and understandable tests of existing and new software pieces.
You understand test as specification that i here to describe what is needed from the
software and also to be able to discuss on this level with non-technical people in case of changes in requirements.
Use 'pytest' for testing.

Here are example-tests how to test different parts of a software piece:
1) Simple test:
---
def test_clean_html_content():
    html_content = "<html><body><article>Test content</article></body></html>"
    result = clean_html_content(html_content)
    assert result.strip() == "Test content"
---

2) Test with variation in input- and output-data but same test code base:
---
@pytest.mark.parametrize("input_html, content_part", [
    ("<html><head></head><body><p>Test content</p></body></html>", "<body><p>Test content</p></body>"),
    ("<html><body><div>Another test</div></body></html>", "<body><div>Another test</div></body>"),
    
    ("<div>Only part of html</div>", "<div>Only part of html</div>"),
    ("Not actually html", "Not actually html"),
])
def test_extract_content_part_from_html(input_html, content_part):
    assert str(extract_content_part_from_html(input_html)) == content_part
---

3) Test with mock:
---
def test_get_documents_from_urls_with_invalid_urls():
    with patch("lib.index.web.get_plain_content_from_url_or_cache") as mock_get_content:
        mock_producer_sink = Mock()
        lib.index.html.get_documents_from_urls(["hp://test.com"], mock_producer_sink)
        mock_get_content.assert_not_called()
        mock_producer_sink.assert_not_called()
---

The user will now give you the code of a method.
You will answer with python-code of multiple test-cases for this method.
"""
def create_test_method(llm: OpenAILike, method_code) -> ChatResponse:
    # system_msg = ChatMessage(role="system", content=sys_msg)
    user_msg = ChatMessage(role="user", content=f"{sys_msg}\n\nPlease create test methods for the following code: ---\n"+method_code+"\n---")
    return llm.chat([user_msg])
    
def get_llm_together(openai_model: str) -> OpenAILike:
    temperature = 0.1
    print(f"About to instanciate LLM {openai_model} using Together.ai ...")
    return OpenAILike(
        model=openai_model,
        api_base="https://api.together.xyz",
        api_key=os.getenv("TOGETHER_AI_KEY"),
        is_chat_model=True,
        is_function_calling_model=True,
        reuse_client=False, # When doing anything with large volumes of async API calls, setting this to false can improve stability.",
        max_retries=10,
        timeout=120,
        temperature=temperature
    )

def get_llm_local_ollama(llm_model: str) -> Ollama:
    api_base_url = f"http://localhost:11434"
    print(f"About to instanciate LLM {llm_model} on {api_base_url} using local Ollama-Instance ...")
    return Ollama(
        model=llm_model, 
        base_url=api_base_url, 
        request_timeout=900, 
        temperature=0.1,
        additional_kwargs={"num_predict": 1024}
    )

def get_llm_openai(openai_model: str) -> OpenAI:
        print(f"About to instanciate LLM {openai_model} using OpenAI ...")
        return OpenAI(
            model=openai_model,
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0.1
        )

def get_method_blocks(file_path):
    with open(file_path, "r") as file:
        source = file.read()
        lines = source.splitlines()

    tree = ast.parse(source)
    method_blocks = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno - 1
            end_line = node.end_lineno
            method_source = "\n".join(lines[start_line:end_line])
            method_blocks.append(method_source)

    return method_blocks

import re
def get_code_parts_from_text(text):
    # Regular expression to match code blocks
    # It looks for the pattern ``` followed by optional language specifier (like python), and then captures everything until the closing ```
    pattern = r"```(?:python)?(.*?)```"
    # re.DOTALL makes the '.' special character match any character at all, including a newline
    matches = re.findall(pattern, text, re.DOTALL)
    # Strip leading/trailing whitespace from each code part
    return [code.strip()+"\n" for code in matches]

def get_method_name_from_code(code):
    # Regular expression to match function definition
    # It looks for the pattern 'def' followed by function name and '()'
    pattern = r"def (\w+)\("
    match = re.search(pattern, code)
    if match:
        # Return the first captured group, which is the function name
        return match.group(1)
    else:
        # Return None or raise an error if no function definition is found
        return None

# def get_method_name_from_code(method_code: str) -> str:
#     return method_code.split("\n")[0].split("def ")[1].split("(")[0]

def to_target_content(method: str, tm_response: ChatResponse):
    method_first_line = method.split("\n")[0]
    result_text = "\n\n# METHOD:\n# "+method_first_line+"\n"

    full_model_response_txt = str(tm_response)
    code_parts = get_code_parts_from_text(full_model_response_txt)
    result_text = result_text + "\n".join(code_parts)
    full_response_as_comment = "\n\n# FULL-RESPONSE:\n# " + "\n".join(["# "+line for line in full_model_response_txt.split("\n")]) + "\n\n"
    return result_text + full_response_as_comment

def write_to_test_file(test_file: str, target_content: str):
    with open(test_file, "a") as file:
        file.write(target_content)

fixed_imports = ["import pytest", "from unittest.mock import Mock, patch", "import json"]
def get_imports(file_path) -> List[str]:
    with open(file_path, "r") as file:
        source = file.read()
        lines = source.splitlines()

    imports = fixed_imports

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            start_line = node.lineno - 1
            end_line = node.end_lineno
            import_source = "\n".join(lines[start_line:end_line])
            imports.append(import_source)
        elif isinstance(node, ast.ImportFrom):
            start_line = node.lineno - 1
            end_line = node.end_lineno
            import_source = "\n".join(lines[start_line:end_line])
            imports.append(import_source)

    return imports

def test_file_from_source_file(source_file: str, test_file_postfix: str) -> str:
    filename = source_file.split("/")[-1]
    test_file_path = "tests/" + source_file.replace(filename, "")
    return test_file_path + "test_" + filename.replace(".py", f"_{test_file_postfix}.py")

def write_imports_to_file_if_not_exists_or_empty(test_file: str, imports: List[str]):
    if not os.path.exists(test_file) or os.path.getsize(test_file) == 0:
        imports = [import_line.strip() for import_line in imports]
        with open(test_file, "w") as file:
            file.write("\n".join(imports))

def get_method_imports(source_file: str, methods: List[str]) -> List[str]:
    imports = []
    method_import_grouped_by_lib_path = {}
    for method in methods:
        method_name = get_method_name_from_code(method)
        lib_path = source_file.replace(".py", "").replace("/", ".")
        if lib_path.startswith("."):
            lib_path = lib_path[1:]
        if lib_path not in method_import_grouped_by_lib_path:
            method_import_grouped_by_lib_path[lib_path] = []
        method_import_grouped_by_lib_path[lib_path].append(method_name)

    for lib_path, method_names in method_import_grouped_by_lib_path.items():
        imports.append(f"from {lib_path} import ({', '.join(method_names)})")
    return imports

def process_source_file(source_file: str, llm, test_file_postfix: str):
        print("\n\nFILE: "+source_file)
        methods = get_method_blocks(source_file)
        if len(methods) == 0:
            print("No methods found in file. Skipping ...")
            return
        target_test_file = test_file_from_source_file(source_file, test_file_postfix)
        os.makedirs(os.path.dirname(target_test_file), exist_ok=True)
        imports = get_imports(source_file)
        imports = imports + get_method_imports(source_file, methods)
        write_imports_to_file_if_not_exists_or_empty(target_test_file, imports)
        for method_code in methods:
            print("\nMETHOD:\n"+method_code+"\n")
            if get_method_name_from_code(method_code).startswith("_"):
                print("Method name starts with '_'. Skipping ...")
                continue
            print("Creating test code via LLM ...\n")
            tm_response = create_test_method(llm, method_code)
            target_content = to_target_content(method_code, tm_response)
            print(f"Target-Content-Start: {target_content[:50]} ...")
            write_to_test_file(target_test_file, target_content)

from concurrent.futures import ThreadPoolExecutor, as_completed
if __name__ == "__main__":
    variant = "ollama"
    test_file_postfix = "ollama5"
    if variant == "together":
         # "codellama/CodeLlama-70b-Python-hf") # "mistralai/Mixtral-8x7B-Instruct-v0.1")
        llm = get_llm_together("codellama/CodeLlama-13b-Python-hf")
    elif variant == "ollama":
        llm = get_llm_local_ollama("codellama:13b") # "codellama:7b-code-q4_K_M")
    elif variant == "openai":
        llm = get_llm_openai("gpt-4-0125-preview") # "gpt-3.5-turbo"
    else:
        raise ValueError(f"Variant {variant} not supported.")
    
    source_files = helper.list_files("lib/", ".py")
    source_files = [file for file in source_files if not "__" in file]
    max_threads = 1
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(process_source_file, file, llm, test_file_postfix) for file in source_files]
        results = [future.result() for future in as_completed(futures)]