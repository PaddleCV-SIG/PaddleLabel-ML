import base64
import io

from PIL import Image
import cv2
import numpy as np


class BaseModel:
    name = "Base Model"
    def __init__(self):
        pass

    def decodeb64(self, img_b64):
        img = img_b64.encode("ascii")
        img = base64.b64decode(img)
        img = Image.open(io.BytesIO(img))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        return img