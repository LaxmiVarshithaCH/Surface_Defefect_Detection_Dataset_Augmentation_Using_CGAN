import os
import json
import datetime


REG_DIR = "registry"

os.makedirs(REG_DIR, exist_ok=True)


VERSION_NAME = "CGAN-Surface-v1.0"


config = {

    "version": VERSION_NAME,

    "date": str(datetime.datetime.now()),

    "dataset": "NEU",

    "img_size": 128,

    "epochs": 300,

    "batch": 64,

    "lr": 1e-4,

    "classes": 6

}


with open(f"{REG_DIR}/config.json","w") as f:
    json.dump(config,f,indent=4)



metrics = {

    "notes": "baseline"

}


with open(f"{REG_DIR}/metrics.json","w") as f:
    json.dump(metrics,f,indent=4)