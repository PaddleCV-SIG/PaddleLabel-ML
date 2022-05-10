# from typing import List
# import numpy as np
# import paddle
# from inference.clicker import Clicker, Click
# from inference.predictor import get_predictor
# import paddle.inference as paddle_infer
# # from models.is_hrnet_model import HRNetModel


# class Predictor:
#     def __init__(self):
#         config = paddle_infer.Config(
#             "static_hrnet18s_cocolvis_model.pdmodel", "static_hrnet18s_cocolvis_model.pdiparams"
#         )
#         config.enable_mkldnn()
#         config.enable_mkldnn_bfloat16()
#         config.switch_ir_optim(True)
#         config.set_cpu_math_library_num_threads(10)
#         self.eiseg = paddle_infer.create_predictor(config)

#         self.predictor_params = {
#             "brs_mode": "NoBRS",
#             "zoom_in_params": {
#                 "skip_clicks": -1,
#                 "target_size": (400, 400),
#                 "expansion_ratio": 1.4,
#             },
#             "predictor_params": {"net_clicks_limit": None, "max_size": 800},
#         }
#         self.predictor = get_predictor(self.eiseg, **self.predictor_params)

#     def run(
#         self,
#         image: np.array,
#         clicker_list: List,
#         pred_thr: float = 0.49,
#         pred_mask: np.array = None,
#     ):
#         clicker = Clicker()
#         self.predictor.set_input_image(image)
#         if pred_mask is not None:
#             pred_mask = paddle.to_tensor(pred_mask[np.newaxis, np.newaxis, :, :])

#         for click_indx in clicker_list:
#             click = Click(is_positive=click_indx[2], coords=(click_indx[1], click_indx[0]))
#             clicker.add_click(click)
#         pred_probs = self.predictor.get_prediction(clicker, pred_mask)
#         output = pred_probs > pred_thr

#         return output, pred_probs


# from flask import Flask, request
# import cv2
# import base64
# import io
# from PIL import Image
# import matplotlib.pyplot as plt

# app = Flask(__name__)

# eiseg = Predictor()

# @app.route("/predict")
# def predict():
#     img = request.json.get("image").encode("ascii")
#     img = base64.b64decode(img)
#     img = Image.open(io.BytesIO(img))
#     img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

#     clicks = request.json.get("clicks")
#     print(clicks)
#     output, pred2 = eiseg.run(img, clicks)

#     cv2.imwrite('out.png', output * 255)

#     return "<p>Hello, World!</p>"


# app.run(host="0.0.0.0", port=17995, debug=True)