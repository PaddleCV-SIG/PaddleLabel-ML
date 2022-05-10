import os.path as osp

from pplabel_ml.model import BaseModel, add_model
from pplabel_ml.util import abort
import cv2
# from pplabel_ml.model.util import copycontent

curr_path = osp.abspath(osp.dirname(__file__))

from typing import List
import numpy as np
import paddle
from .inference.clicker import Clicker, Click
from .inference.predictor import get_predictor
import paddle.inference as paddle_infer

# from models.is_hrnet_model import HRNetModel


class Predictor:
    def __init__(self):
        config = paddle_infer.Config(
            "static_hrnet18s_cocolvis_model.pdmodel",
            "static_hrnet18s_cocolvis_model.pdiparams",
        )

        config.enable_mkldnn()
        config.enable_mkldnn_bfloat16()
        config.switch_ir_optim(True)
        config.set_cpu_math_library_num_threads(10)
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
        pred_thr: float = 0.49,
        pred_mask: np.array = None,
    ):
        clicker = Clicker()
        self.predictor.set_input_image(image)
        if pred_mask is not None:
            pred_mask = paddle.to_tensor(pred_mask[np.newaxis, np.newaxis, :, :])

        for click_indx in clicker_list:
            click = Click(is_positive=click_indx[2], coords=(click_indx[1], click_indx[0]))
            clicker.add_click(click)
        pred_probs = self.predictor.get_prediction(clicker, pred_mask)
        output = pred_probs > pred_thr

        return output, pred_probs


@add_model
class EiSeg(BaseModel):
    name = "EiSeg"

    def __init__(self, param_path=osp.join(curr_path, "ckpt", "best_model")):
        """
        init model

        Args:
            param_path (str, optional): The "best model" path, will load model from this path for inference. Defaults to osp.join(curr_path, "ckpt", "best_model").
        """
        super().__init__(curr_path=curr_path)
        self.model = Predictor()

    def predict(self, req):
        print("req", req["other"]["clicks"], type(req["other"]["clicks"]))
        clicks = req["other"]["clicks"]
        img = self.get_image(req)
        if self.model is None:
            abort("Model is not loaded.")
        output, pred = self.model.run(img, clicks)
        cv2.imwrite("output.png", output*255)

        return output
