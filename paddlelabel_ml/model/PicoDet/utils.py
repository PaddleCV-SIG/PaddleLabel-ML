import time
import os
import yaml

import numpy as np

SUPPORT_MODELS = {
    "YOLO",
    "RCNN",
    "SSD",
    "Face",
    "FCOS",
    "SOLOv2",
    "TTFNet",
    "S2ANet",
    "JDE",
    "FairMOT",
    "DeepSORT",
    "GFL",
    "PicoDet",
    "CenterNet",
    "TOOD",
    "RetinaNet",
    "StrongBaseline",
    "STGCN",
    "YOLOX",
    "PPHGNet",
    "PPLCNet",
}


class Times(object):
    def __init__(self):
        self.time = 0.0
        # start time
        self.st = 0.0
        # end time
        self.et = 0.0

    def start(self):
        self.st = time.time()

    def end(self, repeats=1, accumulative=True):
        self.et = time.time()
        if accumulative:
            self.time += (self.et - self.st) / repeats
        else:
            self.time = (self.et - self.st) / repeats

    def reset(self):
        self.time = 0.0
        self.st = 0.0
        self.et = 0.0

    def value(self):
        return round(self.time, 4)


class Timer(Times):
    def __init__(self, with_tracker=False):
        super(Timer, self).__init__()
        self.with_tracker = with_tracker
        self.preprocess_time_s = Times()
        self.inference_time_s = Times()
        self.postprocess_time_s = Times()
        self.tracking_time_s = Times()
        self.img_num = 0

    def info(self, average=False):
        pre_time = self.preprocess_time_s.value()
        infer_time = self.inference_time_s.value()
        post_time = self.postprocess_time_s.value()
        track_time = self.tracking_time_s.value()

        total_time = pre_time + infer_time + post_time
        if self.with_tracker:
            total_time = total_time + track_time
        total_time = round(total_time, 4)
        print("------------------ Inference Time Info ----------------------")
        print("total_time(ms): {}, img_num: {}".format(total_time * 1000, self.img_num))
        preprocess_time = round(pre_time / max(1, self.img_num), 4) if average else pre_time
        postprocess_time = round(post_time / max(1, self.img_num), 4) if average else post_time
        inference_time = round(infer_time / max(1, self.img_num), 4) if average else infer_time
        tracking_time = round(track_time / max(1, self.img_num), 4) if average else track_time

        average_latency = total_time / max(1, self.img_num)
        qps = 0
        if total_time > 0:
            qps = 1 / average_latency
        print("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency * 1000, qps))
        if self.with_tracker:
            print(
                "preprocess_time(ms): {:.2f}, inference_time(ms): {:.2f}, postprocess_time(ms): {:.2f}, tracking_time(ms): {:.2f}".format(
                    preprocess_time * 1000, inference_time * 1000, postprocess_time * 1000, tracking_time * 1000
                )
            )
        else:
            print(
                "preprocess_time(ms): {:.2f}, inference_time(ms): {:.2f}, postprocess_time(ms): {:.2f}".format(
                    preprocess_time * 1000, inference_time * 1000, postprocess_time * 1000
                )
            )

    def report(self, average=False):
        dic = {}
        pre_time = self.preprocess_time_s.value()
        infer_time = self.inference_time_s.value()
        post_time = self.postprocess_time_s.value()
        track_time = self.tracking_time_s.value()

        dic["preprocess_time_s"] = round(pre_time / max(1, self.img_num), 4) if average else pre_time
        dic["inference_time_s"] = round(infer_time / max(1, self.img_num), 4) if average else infer_time
        dic["postprocess_time_s"] = round(post_time / max(1, self.img_num), 4) if average else post_time
        dic["img_num"] = self.img_num
        total_time = pre_time + infer_time + post_time
        if self.with_tracker:
            dic["tracking_time_s"] = round(track_time / max(1, self.img_num), 4) if average else track_time
            total_time = total_time + track_time
        dic["total_time_s"] = round(total_time, 4)
        return dic


class PredictConfig:
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, "infer_cfg.yml")
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf["arch"]
        self.preprocess_infos = yml_conf["Preprocess"]
        self.min_subgraph_size = yml_conf["min_subgraph_size"]
        self.labels = yml_conf["label_list"]
        self.mask = False
        self.use_dynamic_shape = yml_conf["use_dynamic_shape"]
        if "mask" in yml_conf:
            self.mask = yml_conf["mask"]
        self.tracker = None
        if "tracker" in yml_conf:
            self.tracker = yml_conf["tracker"]
        if "NMS" in yml_conf:
            self.nms = yml_conf["NMS"]
        if "fpn_stride" in yml_conf:
            self.fpn_stride = yml_conf["fpn_stride"]
        if self.arch == "RCNN" and yml_conf.get("export_onnx", False):
            print("The RCNN export model is used for ONNX and it only supports batch_size = 1")
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type
        """
        for support_model in SUPPORT_MODELS:
            if support_model in yml_conf["arch"]:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf["arch"], SUPPORT_MODELS))

    def print_config(self):
        print("-----------  Model Configuration -----------")
        print("%s: %s" % ("Model Arch", self.arch))
        print("%s: " % ("Transform Order"))
        for op_info in self.preprocess_infos:
            print("--%s: %s" % ("transform op", op_info["type"]))
        print("--------------------------------------------")


def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        imgs (list(numpy)): list of images (np.ndarray)
        im_info (list(dict)): list of image info
    Returns:
        inputs (dict): input of model
    """
    inputs = {}

    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs["image"] = np.array((imgs[0],)).astype("float32")
        inputs["im_shape"] = np.array((im_info[0]["im_shape"],)).astype("float32")
        inputs["scale_factor"] = np.array((im_info[0]["scale_factor"],)).astype("float32")
        return inputs

    for e in im_info:
        im_shape.append(np.array((e["im_shape"],)).astype("float32"))
        scale_factor.append(np.array((e["scale_factor"],)).astype("float32"))

    inputs["im_shape"] = np.concatenate(im_shape, axis=0)
    inputs["scale_factor"] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros((im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs["image"] = np.stack(padding_imgs, axis=0)
    return inputs


coco_clsid2catid = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 13,
    12: 14,
    13: 15,
    14: 16,
    15: 17,
    16: 18,
    17: 19,
    18: 20,
    19: 21,
    20: 22,
    21: 23,
    22: 24,
    23: 25,
    24: 27,
    25: 28,
    26: 31,
    27: 32,
    28: 33,
    29: 34,
    30: 35,
    31: 36,
    32: 37,
    33: 38,
    34: 39,
    35: 40,
    36: 41,
    37: 42,
    38: 43,
    39: 44,
    40: 46,
    41: 47,
    42: 48,
    43: 49,
    44: 50,
    45: 51,
    46: 52,
    47: 53,
    48: 54,
    49: 55,
    50: 56,
    51: 57,
    52: 58,
    53: 59,
    54: 60,
    55: 61,
    56: 62,
    57: 63,
    58: 64,
    59: 65,
    60: 67,
    61: 70,
    62: 72,
    63: 73,
    64: 74,
    65: 75,
    66: 76,
    67: 77,
    68: 78,
    69: 79,
    70: 80,
    71: 81,
    72: 82,
    73: 84,
    74: 85,
    75: 86,
    76: 87,
    77: 88,
    78: 89,
    79: 90,
}
