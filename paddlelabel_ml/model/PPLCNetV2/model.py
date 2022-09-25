import os.path as osp
import time
from pathlib import Path

curr_path = osp.abspath(osp.dirname(__file__))
HERE = Path(__file__).parent.absolute()

import paddle
import paddle.inference as paddle_infer

from .operators import *
from .postprocess import Topk
from paddlelabel_ml.model import BaseModel
from paddlelabel_ml.util import abort


class Predictor:
    def __init__(self, model_path: str, param_path: str, use_gpu=False):

        model_path = osp.abspath(model_path)
        param_path = osp.abspath(param_path)

        paddle.disable_static()
        config = paddle_infer.Config(model_path, param_path)
        if not use_gpu:
            config.enable_mkldnn()
            config.enable_mkldnn_bfloat16()
            config.switch_ir_optim(True)
            config.set_cpu_math_library_num_threads(10)
        else:
            config.enable_use_gpu(200, 0)
            # optimize graph and fuse op
            config.switch_ir_optim(True)
        self.predictor = paddle_infer.create_predictor(config)
        self.transforms = Compose(
            [
                ResizeImage(resize_short=256),
                CropImage(size=224),
                NormalizeImage(scale=0.00392157, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToCHWImage(),
            ]
        )
        self.topk = Topk(5, class_id_map_file="labels.txt")

    def preprocess(self, image):
        return self.transforms(image)

    def run(self, images):
        input_names = self.predictor.get_input_names()
        input_tensor = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_tensor = self.predictor.get_output_handle(output_names[0])
        if not isinstance(images, (list,)):
            images = [images]
        for idx in range(len(images)):
            images[idx] = self.preprocess(images[idx])
        input = np.array(images)
        input_tensor.copy_from_cpu(input)
        self.predictor.run()
        output = output_tensor.copy_to_cpu()
        output = self.topk(paddle.to_tensor(output))

        return output


class ClassPretrainNet(BaseModel):
    def __init__(self, model_path: str = None, param_path: str = None):
        """
        init model

        Args:
            model_path (str, optioanl):
            param_path (str, optional):
        """
        super().__init__(curr_path=curr_path)
        if model_path is None:
            model_path = osp.join(curr_path, "ckpt", "inference.pdmodel")
            # model_path = osp.normpath(model_path)
            print("======== model_path ========", model_path)
        else:
            if not osp.exists(model_path):
                abort(f"No model file found at path {model_path}")
        if param_path is None:
            param_path = osp.join(curr_path, "ckpt", "inference.pdiparams")
            print("======== param_path ========", param_path)
        else:
            if not osp.exists(param_path):
                abort(f"No parameter file found at path {param_path}")
        self.model = Predictor(model_path, param_path)
        lines = [l.strip().split(" ") for l in open(HERE / "labels.txt", "r").readlines()]
        self.label_id2name = {int(l[0]): " ".join(l[1:]) for l in lines}

    def predict(self, req):
        img = self.get_image(req)
        if self.model is None:
            abort("Model is not loaded.")
        pred = self.model.run(img)[0]

        predictions = [
            {"label_name": self.label_id2name[idx], "score": score}
            for idx, score in zip(pred["class_ids"], pred["scores"])
        ]
        return predictions


if __name__ == "__main__":
    predictor = Predictor(
        model_path="/Users/haoyuying/Desktop/PPLCNetV2_base_infer/inference.pdmodel",
        param_path="/Users/haoyuying/Desktop/PPLCNetV2_base_infer/inference.pdiparams",
    )
    image = cv2.imread("/Users/haoyuying/Documents/PaddleClas/docs/images/inference_deployment/whl_demo.jpg")
    result = predictor.run(image)
    print(result)
