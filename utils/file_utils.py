# utils/file_utils.py

"""
File I/O utilities for handling JSON, JSONL, and plain text files.

Functions:
- read_jsonl: Load a JSONL file line-by-line into a list of Python objects
- write_jsonl: Save a list of Python objects into a JSONL file
- read_json: Load a JSON file into a Python object
- write_json: Write a Python object into a JSON file
- read_text: Read plain text from a file
- write_text: Write plain text to a file
"""

import json


def read_jsonl(filename):
    """
    Reads a JSONL (JSON Lines) file into a list of Python dicts/lists.

    Args:
        filename (str): Path to the input .jsonl file.

    Returns:
        list: Parsed list of JSON objects.
    """
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[read_jsonl] Skipping invalid JSON line: {e}")
    return data


def write_jsonl(data_list, filename):
    """
    Writes a list of Python objects to a JSONL file.

    Args:
        data_list (list): List of dictionaries or lists to write.
        filename (str): Output file path.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item) + "\n")


def read_json(filename):
    """
    Reads a standard JSON file.

    Args:
        filename (str): Path to the JSON file.

    Returns:
        dict or list: Parsed JSON object.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, filename):
    """
    Writes a dictionary or list to a JSON file.

    Args:
        data (dict or list): Data to write.
        filename (str): Output file path.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_text(filename):
    """
    Reads plain text from a file.

    Args:
        filename (str): Path to the file.

    Returns:
        str: Entire text content.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def write_text(text, filename):
    """
    Writes plain text to a file.

    Args:
        text (str): The string to write.
        filename (str): Output file path.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
