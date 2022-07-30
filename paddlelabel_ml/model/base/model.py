import base64
import io
import os
import os.path as osp

from PIL import Image
import cv2
import numpy as np

from paddlelabel_ml.util import abort

import matplotlib.pyplot as plt


class BaseModel:
    name = "Base Model"

    def __init__(self, curr_path: str):
        print("curr_path", curr_path)
        self.params = None
        self.ckpt_path = osp.join(curr_path, "ckpt")

        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(osp.join(self.ckpt_path, "run"), exist_ok=True)

        runs = os.listdir(osp.join(self.ckpt_path, "run"))
        runs = [int(n) for n in runs]
        if len(runs) == 0:
            self.output_path = osp.join(self.ckpt_path, "run", "1")
        else:
            self.output_path = osp.join(self.ckpt_path, "run", str(max(runs) + 1))

    def get_image(self, req):
        if req["format"] == "b64":
            return self.decodeb64(req["img"])

        if req["format"] == "path":
            img = cv2.cvtColor(cv2.imread(req["img"]), cv2.COLOR_BGR2RGB)
            return img

    def decodeb64(self, img_b64):
        img = img_b64.encode("ascii")
        img = base64.b64decode(img)
        img = Image.open(io.BytesIO(img))
        img = img.convert("RGB")
        img = np.asarray(img)
        print("Image shape:", img.shape)

        # plt.imshow(img)
        # plt.savefig('/pwd/test.png')

        return img

    def pretrain_check(
        self,
        data_dir,
        requirements={"multiple label": False, "files": ["labels.txt", "train_list.txt", "val_list.txt"]},
    ):

        for fn in requirements["files"]:
            path = osp.join(data_dir, fn)
            if not osp.exists(path):
                abort(f"Required file {fn} not present", 404)
            lines = open(path, "r").readlines()
            lines = [l.strip() for l in lines if len(l.strip()) != 0]
            if len(lines) == 0:
                abort(f"Required file {fn} empty", 404)
