from src.monitor.log_usage import log_usages
import torch
import numpy as np
import base64
import cv2

from fastapi import FastAPI
from pydantic import BaseModel

from src.generator_cgan_surface import Generator
from src.config import load_config


cfg = load_config()

Z_DIM = cfg["z_dim"]
CLASSES = cfg["classes"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator().to(device)

G.load_state_dict(
    torch.load(cfg["checkpoint"], map_location=device)
)

G.eval()


app = FastAPI()


class Req(BaseModel):

    defect_type: int
    count: int


def generate(cls):

    z = torch.randn(1, Z_DIM).to(device)

    lab = torch.tensor([cls]).to(device)

    with torch.no_grad():

        img = G(z, lab)[0]

    img = img.cpu().permute(1,2,0).numpy()

    img = (img + 1) / 2

    img = (img * 255).astype(np.uint8)

    _, buf = cv2.imencode(".png", img)

    b64 = base64.b64encode(buf).decode()

    return b64


@app.post("/generate_defects")
def gen(req: Req):

    log_usage(
        source="api",
        cls=CLASSES[req.defect_type],
        n=req.count,
    )

    out = []

    for i in range(req.count):

        out.append(generate(req.defect_type))

    return {"images": out}


@app.get("/defect_types")

def types():

    return CLASSES