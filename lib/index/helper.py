from datetime import datetime
import os

def cur_simple_date_time_sec() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def list_files(directory: str, file_ext_filter: str = None):
    """
    List files in a directory recursively, filtering by file extensions.

    :param directory: The starting directory to search
    :param extensions: List of file extensions to include (without leading dot)
    :return: A list of file paths
    """
    files_list = []
    for root, _, files in os.walk(directory):
        for file in files:
                files_list.append(os.path.join(root, file))
    if file_ext_filter:
        files_list = [f for f in files_list if f.endswith(file_ext_filter)]
    return files_list

