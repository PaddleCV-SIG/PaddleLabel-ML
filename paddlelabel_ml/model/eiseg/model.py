import os.path as osp
import time

import cv2
import numpy as np
import paddle
import paddle.inference as paddle_infer

from paddlelabel_ml.model import BaseModel
from paddlelabel_ml.util import abort, use_gpu
from typing import List
from .inference.clicker import Clicker, Click
from .inference.predictor import get_predictor

curr_path = osp.abspath(osp.dirname(__file__))


class Predictor:
    def __init__(self, model_path: str, param_path: str):
        model_path = osp.abspath(model_path)
        param_path = osp.abspath(param_path)

        paddle.disable_static()
        config = paddle_infer.Config(model_path, param_path)

        config.enable_mkldnn()
        config.enable_mkldnn_bfloat16()
        config.switch_ir_optim(True)
        config.set_cpu_math_library_num_threads(10)
        if use_gpu:
            config.enable_use_gpu(100)
        self.eiseg = paddle_infer.create_predictor(config)

        self.predictor_params = {
            "brs_mode": "NoBRS",
            "zoom_in_params": {
                "skip_clicks": -1,
                "target_size": (400, 400),
                "expansion_ratio": 1.4,
            },
            "predictor_params": {"net_clicks_limit": None, "max_size": 800},
        }
        self.predictor = get_predictor(self.eiseg, **self.predictor_params)

    def run(
        self,
        image: np.array,
        clicker_list: List,
        pred_mask: np.array = None,
    ):

        clicker = Clicker()
        self.predictor.set_input_image(image)
        if pred_mask is not None:
            pred_mask = paddle.to_tensor(pred_mask[np.newaxis, np.newaxis, :, :])

        for click_indx in clicker_list:
            click = Click(is_positive=click_indx[2], coords=(click_indx[1], click_indx[0]))
            clicker.add_click(click)

        tic = time.time()
        pred_probs = self.predictor.get_prediction(clicker, pred_mask)
        print(f"======== Inference on paddle took {time.time() - tic} s")

        return pred_probs


class EISeg(BaseModel):
    name = "EISeg"

    def __init__(self, model_path: str = None, param_path: str = None):
        """
        init model

        Args:
            model_path (str, optioanl):
            param_path (str, optional):
        """
        super().__init__(curr_path=curr_path)
        if model_path is None:
            model_path = osp.join(curr_path, "ckpt", "static_hrnet18_ocr64_cocolvis.pdmodel")
            # model_path = osp.normpath(model_path)
            print("======== model_path ========", model_path)
        else:
            if not osp.exists(model_path):
                abort(f"No model file found at path {model_path}")
        if param_path is None:
            param_path = osp.join(curr_path, "ckpt", "static_hrnet18_ocr64_cocolvis.pdiparams")
            print("======== param_path ========", param_path)
        else:
            if not osp.exists(param_path):
                abort(f"No parameter file found at path {param_path}")
        self.model = Predictor(model_path, param_path)

    def predict(self, req):
        print("req", req["other"]["clicks"], type(req["other"]["clicks"]))
        clicks = req["other"]["clicks"]
        print("clicks", clicks)
        img = self.get_image(req)

        if self.model is None:
            abort("Model is not loaded.")
        pred = self.model.run(img, clicks)
        # pred_probs = np.swapaxes(pred_probs, 0, 1)

        pred = pred.tolist()
        pred = [[round(n, 2) for n in line] for line in pred]

        return pred
