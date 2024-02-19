import pytest
from unittest.mock import Mock, patch
import json
from datetime import datetime
import os
from lib.index.helper import (cur_simple_date_time_sec, list_files)

# METHOD:
# def cur_simple_date_time_sec() -> str:
import pytest
from datetime import datetime, timedelta

def test_cur_simple_date_time_sec():
    # Test that the current date and time is returned in the expected format
    assert cur_simple_date_time_sec() == datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def test_cur_simple_date_time_sec_with_offset():
    # Test that the current date and time is returned in the expected format with an offset
    assert cur_simple_date_time_sec(timedelta(hours=1)) == (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d_%H-%M-%S")

def test_cur_simple_date_time_sec_with_negative_offset():
    # Test that the current date and time is returned in the expected format with a negative offset
    assert cur_simple_date_time_sec(timedelta(hours=-1)) == (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d_%H-%M-%S")


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `cur_simple_date_time_sec` method:
# ```python
# import pytest
# from datetime import datetime, timedelta
# 
# def test_cur_simple_date_time_sec():
#     # Test that the current date and time is returned in the expected format
#     assert cur_simple_date_time_sec() == datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# 
# def test_cur_simple_date_time_sec_with_offset():
#     # Test that the current date and time is returned in the expected format with an offset
#     assert cur_simple_date_time_sec(timedelta(hours=1)) == (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d_%H-%M-%S")
# 
# def test_cur_simple_date_time_sec_with_negative_offset():
#     # Test that the current date and time is returned in the expected format with a negative offset
#     assert cur_simple_date_time_sec(timedelta(hours=-1)) == (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d_%H-%M-%S")
# ```



# METHOD:
# def list_files(directory: str, file_ext_filter: str = None):
def test_no_files(tmpdir):
    directory = tmpdir.mkdir("test")
    assert list_files(directory) == []

def test_with_files(tmpdir):
    directory = tmpdir.mkdir("test")
    file1 = directory.join("file1.txt")
    file2 = directory.join("file2.txt")
    assert list_files(directory) == [file1, file2]

def test_filter_by_extension(tmpdir):
    directory = tmpdir.mkdir("test")
    file1 = directory.join("file1.txt")
    file2 = directory.join("file2.txt")
    assert list_files(directory, "txt") == [file1]

def test_absolute_paths(tmpdir):
    directory = tmpdir.mkdir("test")
    file1 = directory.join("file1.txt")
    file2 = directory.join("file2.txt")
    assert all([os.path.isabs(f) for f in list_files(directory)])

def test_non_existent_directory():
    with pytest.raises(FileNotFoundError):
        list_files("non-existent-directory")


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `list_files` method:
# 
# 1. Test that the method returns an empty list when there are no files in the directory:
# ```python
# def test_no_files(tmpdir):
#     directory = tmpdir.mkdir("test")
#     assert list_files(directory) == []
# ```
# 2. Test that the method returns a list of file paths when there are files in the directory:
# ```python
# def test_with_files(tmpdir):
#     directory = tmpdir.mkdir("test")
#     file1 = directory.join("file1.txt")
#     file2 = directory.join("file2.txt")
#     assert list_files(directory) == [file1, file2]
# ```
# 3. Test that the method filters files by extension when a filter is provided:
# ```python
# def test_filter_by_extension(tmpdir):
#     directory = tmpdir.mkdir("test")
#     file1 = directory.join("file1.txt")
#     file2 = directory.join("file2.txt")
#     assert list_files(directory, "txt") == [file1]
# ```
# 4. Test that the method returns a list of absolute paths:
# ```python
# def test_absolute_paths(tmpdir):
#     directory = tmpdir.mkdir("test")
#     file1 = directory.join("file1.txt")
#     file2 = directory.join("file2.txt")
#     assert all([os.path.isabs(f) for f in list_files(directory)])
# ```
# 5. Test that the method raises an error when the directory does not exist:
# ```python
# def test_non_existent_directory():
#     with pytest.raises(FileNotFoundError):
#         list_files("non-existent-directory")
# ```

