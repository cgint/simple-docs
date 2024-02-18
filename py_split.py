import ast
from lib.index import helper
import os
from llama_index.llms.openai_like import OpenAILike

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
    system_msg = ChatMessage(role="system", content=sys_msg)
    user_msg = ChatMessage(role="user", content="Creating test method for: ---\n"+method_code+"\n---")
    return llm.chat([system_msg, user_msg])
    
def get_llm():
    openai_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
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

def to_target_content(method, tm_response):
    method_first_line = method.split("\n")[0]
    result_text = "# METHOD: "+method_first_line+"\n"

    # When there is content between '```python' and '```' the part between them is returned
    respnose_txt = str(tm_response)
    if "```python" in respnose_txt and "```" in respnose_txt:
        result_text = result_text + respnose_txt.split("```python")[1].split("```")[0]
    else:
        # return the full content as comment - each line starting with '# '
        result_text = result_text + "\n".join(["# "+line for line in respnose_txt.split("\n")])
    return result_text

def write_to_test_file(source_file_path: str, target_content: str):
    filename = source_file_path.split("/")[-1]
    test_file_path = "lib/" + source_file_path.replace(filename, "")
    test_file = test_file_path + "test_" + filename
    with open(test_file, "a") as file:
        file.write(target_content)

if __name__ == "__main__":
    source_files = helper.list_files("lib/", ".py")
    llm = get_llm()
    for source_file in source_files:
        print("\n\nFILE: "+source_file)
        methods = get_method_blocks(source_file)
        for method_code in methods:
            print("\nMETHOD:")
            print(method_code+"\n")
            print("Creating test code via LLM ...\n")
            tm_response = create_test_method(llm, method_code)
            target_content = to_target_content(tm_response)
            print(f"Target-Content-Start: {target_content[:50]} ...")
            # write to test file
            write_to_test_file(source_file, target_content)