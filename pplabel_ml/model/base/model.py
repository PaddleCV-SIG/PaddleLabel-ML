import base64
import io
import os
import os.path as osp

from PIL import Image
import cv2
import numpy as np


class BaseModel:
    name = "Base Model"
    def __init__(self, curr_path:str):
        self.ckpt_path = osp.join(curr_path, "ckpt")
        
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(osp.join(self.ckpt_path, "run"), exist_ok=True)
        
        runs = os.listdir(osp.join(self.ckpt_path, "run"))
        runs = [int(n) for n in runs]
        if len(runs) == 0:
            self.output_path= osp.join(self.ckpt_path, "run", "1")
        else:
            self.output_path=osp.join(self.ckpt_path, "run", str(max(runs) + 1))

    def decodeb64(self, img_b64):
        img = img_b64.encode("ascii")
        img = base64.b64decode(img)
        img = Image.open(io.BytesIO(img))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        return img
    