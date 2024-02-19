import pytest
from unittest.mock import Mock, patch
import json
from lib import constants
from lib.index.helper import cur_simple_date_time_sec
from traceback import format_exception
from lib.index.error_helper import (write_error_to_file)

# METHOD:
# def write_error_to_file(e: Exception, msg: str = None):
def test_write_error_to_file_with_exception(mocker):
    mocker.patch("traceback.format_exception", return_value="Traceback (most recent call last):\n  File \"test.py\", line 10, in <module>\n    raise Exception('Test exception')\nException: Test exception")
    with open(constants.error_file, "a") as f:
        write_error_to_file(Exception("Test exception"), msg="Test message")
        assert f.read() == "========================================================================\n" \
                           "================= 2023-01-01T00:00:00Z ==================\n" \
                           "========================================================================\n" \
                           "Test message\n" \
                           "Trace: Traceback (most recent call last):\n  File \"test.py\", line 10, in <module>\n    raise Exception('Test exception')\nException: Test exception\n" \
                           "========================================================================\n"

def test_write_error_to_file_without_exception(mocker):
    mocker.patch("traceback.format_exception", return_value="")
    with open(constants.error_file, "a") as f:
        write_error_to_file(None)
        assert f.read() == ""

def test_write_error_to_file_with_custom_message(mocker):
    mocker.patch("traceback.format_exception", return_value="")
    with open(constants.error_file, "a") as f:
        write_error_to_file(None, msg="Test message")
        assert f.read() == "========================================================================\n" \
                           "================= 2023-01-01T00:00:00Z ==================\n" \
                           "========================================================================\n" \
                           "Test message\n" \
                           "Trace: \n" \
                           "========================================================================\n"

def test_write_error_to_file_with_custom_date_and_time(mocker):
    mocker.patch("traceback.format_exception", return_value="")
    with open(constants.error_file, "a") as f:
        write_error_to_file(None, msg="Test message", date=datetime.datetime(2023, 1, 1, 12, 0, 0))
        assert f.read() == "========================================================================\n" \
                           "================= 2023-01-01T12:00:00Z ==================\n" \
                           "========================================================================\n" \
                           "Test message\n" \
                           "Trace: \n" \
                           "========================================================================\n"


# FULL-RESPONSE:
# # assistant: 
# Here are some test cases for the `write_error_to_file` method:
# 
# 1. Test that the error message is written to the file correctly when an exception occurs:
# ```python
# def test_write_error_to_file_with_exception(mocker):
#     mocker.patch("traceback.format_exception", return_value="Traceback (most recent call last):\n  File \"test.py\", line 10, in <module>\n    raise Exception('Test exception')\nException: Test exception")
#     with open(constants.error_file, "a") as f:
#         write_error_to_file(Exception("Test exception"), msg="Test message")
#         assert f.read() == "========================================================================\n" \
#                            "================= 2023-01-01T00:00:00Z ==================\n" \
#                            "========================================================================\n" \
#                            "Test message\n" \
#                            "Trace: Traceback (most recent call last):\n  File \"test.py\", line 10, in <module>\n    raise Exception('Test exception')\nException: Test exception\n" \
#                            "========================================================================\n"
# ```
# 2. Test that the error message is not written to the file when no exception occurs:
# ```python
# def test_write_error_to_file_without_exception(mocker):
#     mocker.patch("traceback.format_exception", return_value="")
#     with open(constants.error_file, "a") as f:
#         write_error_to_file(None)
#         assert f.read() == ""
# ```
# 3. Test that the error message is written to the file correctly when a custom message is provided:
# ```python
# def test_write_error_to_file_with_custom_message(mocker):
#     mocker.patch("traceback.format_exception", return_value="")
#     with open(constants.error_file, "a") as f:
#         write_error_to_file(None, msg="Test message")
#         assert f.read() == "========================================================================\n" \
#                            "================= 2023-01-01T00:00:00Z ==================\n" \
#                            "========================================================================\n" \
#                            "Test message\n" \
#                            "Trace: \n" \
#                            "========================================================================\n"
# ```
# 4. Test that the error message is written to the file correctly when a custom date and time are provided:
# ```python
# def test_write_error_to_file_with_custom_date_and_time(mocker):
#     mocker.patch("traceback.format_exception", return_value="")
#     with open(constants.error_file, "a") as f:
#         write_error_to_file(None, msg="Test message", date=datetime.datetime(2023, 1, 1, 12, 0, 0))
#         assert f.read() == "========================================================================\n" \
#                            "================= 2023-01-01T12:00:00Z ==================\n" \
#                            "========================================================================\n" \
#                            "Test message\n" \
#                            "Trace: \n" \
#                            "========================================================================\n"
# ```

