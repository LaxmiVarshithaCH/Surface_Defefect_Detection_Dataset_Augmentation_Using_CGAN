import json

CONFIG_PATH = "configs/config.json"


def load_config():

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    return cfg