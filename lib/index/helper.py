from datetime import datetime
import os

def cur_simple_date_time_sec() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def list_files(directory):
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
    return files_list