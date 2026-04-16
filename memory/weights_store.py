import json
from pathlib import Path
import threading

WEIGHT_FILE = Path("memory/weights.json")
_lock = threading.Lock()


def load_weights():
    if WEIGHT_FILE.exists():
        try:
           return json.loads(WEIGHT_FILE.read_text(encoding="utf-8"))
        except:
            return {}
    return {}


def save_weights(data):
    WEIGHT_FILE.parent.mkdir(exist_ok=True)
    WEIGHT_FILE.write_text(json.dumps(data, indent=2))


def update_weight(doc_id, pages):
    key = f"{doc_id}_{pages}"

    with _lock:
        data = load_weights()
        data[key] = data.get(key, 0) + 1
        save_weights(data)