# ui/utils.py
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def list_datasets():
    """
    Return folders like data_1, data_2, data_3 located at project root
    """
    return sorted([
        p for p in ROOT.iterdir()
        if p.is_dir() and p.name.startswith("data_")
    ])

def load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def load_csv(path):
    return pd.read_csv(path) if path.exists() else None
