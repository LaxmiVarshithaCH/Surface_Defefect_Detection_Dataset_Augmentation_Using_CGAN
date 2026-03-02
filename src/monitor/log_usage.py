import json
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_PATH = BASE_DIR / "registry" / "usage_log.json"


def log_usage(source, cls, n):

    if not LOG_PATH.exists():

        LOG_PATH.parent.mkdir(exist_ok=True)

        with open(LOG_PATH, "w") as f:
            json.dump({"requests": []}, f)

    with open(LOG_PATH, "r") as f:

        data = json.load(f)

    if isinstance(data, list):

        requests = data

    else:

        requests = data.get("requests", [])

    entry = {
        "time": datetime.now().isoformat(),
        "source": source,
        "class": cls,
        "n": n,
    }

    requests.append(entry)

    new_data = {"requests": requests}

    with open(LOG_PATH, "w") as f:

        json.dump(new_data, f, indent=2)