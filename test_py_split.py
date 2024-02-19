import pytest
from py_split import get_code_parts_from_text, get_method_blocks, get_method_blocks_from_source, get_method_imports, get_method_name_from_code

def test_get_one_code_part_from_text():
    the_code = """import pytest
def test_get_code_parts_from_text():
    assert get_code_parts_from_text(text) == [""]
"""
    text = f"""
blabla-text
text```python{the_code}```
"""
    parts = get_code_parts_from_text(text)
    print("\n\n ==== PARTS: ====\n"+str(parts)+"\n=================\n\n")
    print("\n\n ==== THECD: ====\n"+str(the_code)+"\n=================\n\n")
    assert parts == [the_code]

def test_get_multiple_code_parts_from_text():
    the_code1 = """import pytest
def test_get_code_parts_from_text1():
    assert get_code_parts_from_text(text1) == [""]
"""
    the_code2 = """import pytest
def test_get_something_from_text2():
    assert get_code_parts_from_text(text2) == [""]
"""
    the_code3 = """import pytest
def test_get_code_parts_from_text3():
    assert get_code_parts_from_text(text3) == [""]
"""
    the_code4 = """import pytest
def test_get_code_parts_from_text4():
    assert get_code_parts_from_text(text4) == [""]
"""
    the_code5 = """import pytest
def test_get_code_parts_from_text5():
    assert get_code_parts_from_text(text5) == [""]
"""
    text = f"""
blabla-text
text```python{the_code1}```

even more bla bla
text```{the_code2}```

blabla-text
text```python{the_code3}```

even more bla bla
text```{the_code4}```

even more bla bla
text```{the_code5}```

and in the end
some more bla bla

"""
    parts = get_code_parts_from_text(text)
    expected_parts = [the_code1, the_code2, the_code3, the_code4, the_code5]
    print("\n\n ==== PARTS: ====\n"+str(parts)+"\n=================\n\n")
    print("\n\n ==== THECD: ====\n"+str(expected_parts)+"\n=================\n\n")
    assert len(parts) == len(expected_parts)
    assert parts == expected_parts

def test_get_no_code_parts_from_text():
    llm_response ="""assistant: 
Here are some test cases for the `wrap_in_sub_question_engine` method:

1. Test that the returned query engine is an instance of `SubQuestionQueryEngine`:
```python
def test_returned_query_engine_is_instance_of_sub_question_engine():
    query_engine = RetrieverQueryEngine()
    wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
    assert isinstance(wrapped_query_engine, SubQuestionQueryEngine)
```
2. Test that the `use_async` parameter is set to `True`:
```python
def test_use_async_is_set_to_true():
    query_engine = RetrieverQueryEngine()
    wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
    assert wrapped_query_engine.use_async is True
```
3. Test that the `guidance_llm` parameter is set to an instance of `OpenAI`:
```python
def test_guidance_llm_is_set_to_openai():
    query_engine = RetrieverQueryEngine()
    wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
    assert isinstance(wrapped_query_engine.guidance_llm, OpenAI)
```
4. Test that the `query_engine_tools` parameter is set to a list containing an instance of `QueryEngineTool`:
```python
def test_query_engine_tools_is_set_to_list_containing_query_engine_tool():
    query_engine = RetrieverQueryEngine()
    wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
    assert isinstance(wrapped_query_engine.query_engine_tools[0], QueryEngineTool)
```
5. Test that the `name` parameter of the `QueryEngineTool` instance in the `query_engine_tools` list is set to "hybrid_query_engine":
```python
def test_query_engine_tool_name_is_set_to_hybrid_query_engine():
    query_engine = RetrieverQueryEngine()
    wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
    assert wrapped_query_engine.query_engine_tools[0].name == "hybrid_query_engine"
```
6. Test that the `description` parameter of the `QueryEngineTool` instance in the `query_engine_tools` list is set to a string containing the text "Provides information about everything the user might ask":
```python
def test_query_engine_tool_description_is_set_to_string_containing_text():
    query_engine = RetrieverQueryEngine()
    wrapped_query_engine = wrap_in_sub_question_engine(query_engine)
    assert wrapped_query_engine.query_engine_tools[0].description == "Provides information about everything the user might ask"
```
"""
    parts = get_code_parts_from_text(llm_response)
    print("\n\n ==== PARTS: ====\n"+str(parts)+"\n=================\n\n")
    assert len(parts) == 6

def test_get_method_name_from_code():
    code = """import pytest

def test_get_code_parts_from_text(param1):
    assert get_code_parts_from_text(text) == [""]
"""
    method_name = get_method_name_from_code(code)
    print("\n\n ==== METHOD NAME: ====\n"+str(method_name)+"\n=================\n\n")
    assert method_name == "test_get_code_parts_from_text"

@pytest.mark.parametrize("source_file", [
    ("/path/to/source/file.py"),
    ("path/to/source/file.py")
])
def test_get_method_imports(source_file: str):
    methods = [
        """def test_method_1():
    pass""",
        """def test_method_2():
    pass""",
        """def test_method_3():
    pass"""
    ]

    expected_imports = [
        "from path.to.source.file import (test_method_1, test_method_2, test_method_3)"
    ]

    imports = get_method_imports(source_file, methods)
    assert imports == expected_imports

def test_get_method_blocks_ignore_class_methods():
    source_code = '''
class SomeClass:
    def method_inside_class(self):
        pass

def method_outside_class():
    pass
    '''
    expected_methods = ['def method_outside_class():\n    pass\n']
    assert get_method_blocks_from_source(source_code) == expected_methods
