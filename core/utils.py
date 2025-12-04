import os
import json


def make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
