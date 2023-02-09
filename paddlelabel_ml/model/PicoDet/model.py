import os
import os.path as osp
import json
from pathlib import Path
from functools import reduce

curr_path = osp.abspath(osp.dirname(__file__))
HERE = Path(__file__).parent.absolute()

import cv2
import numpy as np
import math
from paddle.inference import Config
from paddle.inference import create_predictor

from .utils import Timer, PredictConfig, create_inputs, coco_clsid2catid
from .preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride, LetterBoxResize, Pad, decode_image
from .postprocess import PicoDetPostProcess
from .visualize import visualize
from paddlelabel_ml.model import BaseModel
from paddlelabel_ml.util import abort, use_gpu


def load_predictor(
    model_dir,
    run_mode="paddle",
    batch_size=1,
    use_gpu=False,
    min_subgraph_size=3,
    use_dynamic_shape=False,
    trt_min_shape=1,
    trt_max_shape=1280,
    trt_opt_shape=640,
    trt_calib_mode=False,
    cpu_threads=1,
    enable_mkldnn=False,
    enable_mkldnn_bfloat16=False,
    delete_shuffle_pass=False,
):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16/trt_int8)
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT.
                                    Used by action model.
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need device == 'GPU'.
    """

    infer_model = os.path.join(model_dir, "model.pdmodel")
    infer_params = os.path.join(model_dir, "model.pdiparams")
    if not os.path.exists(infer_model):
        infer_model = os.path.join(model_dir, "inference.pdmodel")
        infer_params = os.path.join(model_dir, "inference.pdiparams")
        if not os.path.exists(infer_model):
            raise ValueError("Cannot find any inference model in dir: {},".format(model_dir))
    config = Config(infer_model, infer_params)
    if use_gpu:
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
        print("using gpu:", use_gpu)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        if enable_mkldnn:
            try:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
                if enable_mkldnn_bfloat16:
                    config.enable_mkldnn_bfloat16()
            except Exception as e:
                print("The current environment does not support `mkldnn`, so disable mkldnn.")
                pass

    precision_map = {
        "trt_int8": Config.Precision.Int8,
        "trt_fp32": Config.Precision.Float32,
        "trt_fp16": Config.Precision.Half,
    }
    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=(1 << 25) * batch_size,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[run_mode],
            use_static=False,
            use_calib_mode=trt_calib_mode,
        )

        if use_dynamic_shape:
            min_input_shape = {"image": [batch_size, 3, trt_min_shape, trt_min_shape]}
            max_input_shape = {"image": [batch_size, 3, trt_max_shape, trt_max_shape]}
            opt_input_shape = {"image": [batch_size, 3, trt_opt_shape, trt_opt_shape]}
            config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape, opt_input_shape)
            print("trt set dynamic shape done!")

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    if delete_shuffle_pass:
        config.delete_pass("shuffle_channel_detect_pass")
    predictor = create_predictor(config)
    return predictor, config


class Detector(object):
    """
    Args:
        pred_config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
        enable_mkldnn_bfloat16 (bool): whether to turn on mkldnn bfloat16
        output_dir (str): The path of output
        threshold (float): The threshold of score for visualization
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT.
                                    Used by action model.
    """

    def __init__(
        self,
        model_dir,
        use_gpu=use_gpu,
        run_mode="paddle",
        batch_size=1,
        trt_min_shape=1,
        trt_max_shape=1280,
        trt_opt_shape=640,
        trt_calib_mode=False,
        cpu_threads=1,
        enable_mkldnn=False,
        enable_mkldnn_bfloat16=False,
        output_dir="output",
        threshold=0.5,
        delete_shuffle_pass=False,
    ):
        self.pred_config = self.set_config(model_dir)

        self.predictor, self.config = load_predictor(
            model_dir,
            run_mode=run_mode,
            batch_size=batch_size,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            use_gpu=use_gpu,
            use_dynamic_shape=self.pred_config.use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            enable_mkldnn_bfloat16=enable_mkldnn_bfloat16,
            delete_shuffle_pass=delete_shuffle_pass,
        )
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.threshold = threshold

    def set_config(self, model_dir):
        return PredictConfig(model_dir)

    def preprocess(self, images):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop("type")
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []
        for im_path in images:
            im, im_info = preprocess(im_path, preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            if input_names[i] == "x":
                input_tensor.copy_from_cpu(inputs["image"])
            else:
                input_tensor.copy_from_cpu(inputs[input_names[i]])

        return inputs

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        np_boxes_num = result["boxes_num"]
        if sum(np_boxes_num) <= 0:
            print("[WARNNING] No object detected.")
            result = {"boxes": np.zeros([0, 6]), "boxes_num": [0]}
        result = {k: v for k, v in result.items() if v is not None}
        return result

    def filter_box(self, result, threshold):
        np_boxes_num = result["boxes_num"]
        boxes = result["boxes"]
        start_idx = 0
        filter_boxes = []
        filter_num = []
        for i in range(len(np_boxes_num)):
            boxes_num = np_boxes_num[i]
            boxes_i = boxes[start_idx : start_idx + boxes_num, :]
            idx = boxes_i[:, 1] > threshold
            filter_boxes_i = boxes_i[idx, :]
            filter_boxes.append(filter_boxes_i)
            filter_num.append(filter_boxes_i.shape[0])
            start_idx += boxes_num
        boxes = np.concatenate(filter_boxes)
        filter_num = np.array(filter_num)
        filter_res = {"boxes": boxes, "boxes_num": filter_num}
        return filter_res

    def predict(self, repeats=1):
        """
        Args:
            repeats (int): repeats number for prediction
        Returns:
            result (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's result include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        """
        # model prediction
        np_boxes, np_masks = None, None
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            boxes_num = self.predictor.get_output_handle(output_names[1])
            np_boxes_num = boxes_num.copy_to_cpu()
            if self.pred_config.mask:
                masks_tensor = self.predictor.get_output_handle(output_names[2])
                np_masks = masks_tensor.copy_to_cpu()
        result = dict(boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
        return result

    def merge_batch_result(self, batch_result):
        if len(batch_result) == 1:
            return batch_result[0]
        res_key = batch_result[0].keys()
        results = {k: [] for k in res_key}
        for res in batch_result:
            for k, v in res.items():
                results[k].append(v)
        for k, v in results.items():
            if k not in ["masks", "segm"]:
                results[k] = np.concatenate(v)
        return results

    def get_timer(self):
        return self.det_times

    def run(self, images, run_benchmark=False, repeats=1, visual=False):
        if not isinstance(images, (list,)):
            images = [images]
        batch_loop_cnt = math.ceil(float(len(images)) / self.batch_size)
        results = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(images))
            batch_image_list = images[start_index:end_index]
            if run_benchmark:
                # preprocess
                inputs = self.preprocess(batch_image_list)  # warmup
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                result = self.predict(repeats=50)  # warmup
                self.det_times.inference_time_s.start()
                result = self.predict(repeats=repeats)
                self.det_times.inference_time_s.end(repeats=repeats)

                # postprocess
                result_warmup = self.postprocess(inputs, result)  # warmup
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += len(batch_image_list)

                # cm, gm, gu = get_current_memory_mb()
                # self.cpu_mem += cm
                # self.gpu_mem += gm
                # self.gpu_util += gu
            else:
                # preprocess
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                self.det_times.inference_time_s.start()
                result = self.predict()
                self.det_times.inference_time_s.end()

                # postprocess
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += len(batch_image_list)

                if visual:
                    visualize(
                        batch_image_list,
                        result,
                        self.pred_config.labels,
                        output_dir=self.output_dir,
                        threshold=self.threshold,
                    )
            results.append(result)
            # print("Test iter {}".format(i))

        results = self.merge_batch_result(results)
        return results

    def save_coco_results(self, image_list, results, use_coco_category=False):
        bbox_results = []
        mask_results = []
        idx = 0
        print("Start saving coco json files...")
        for i, box_num in enumerate(results["boxes_num"]):
            file_name = os.path.split(image_list[i])[-1]
            if use_coco_category:
                img_id = int(os.path.splitext(file_name)[0])
            else:
                img_id = i

            if "boxes" in results:
                boxes = results["boxes"][idx : idx + box_num].tolist()
                bbox_results.extend(
                    [
                        {
                            "image_id": img_id,
                            "category_id": coco_clsid2catid[int(box[0])] if use_coco_category else int(box[0]),
                            "file_name": file_name,
                            "bbox": [box[2], box[3], box[4] - box[2], box[5] - box[3]],  # xyxy -> xywh
                            "score": box[1],
                        }
                        for box in boxes
                    ]
                )

            if "masks" in results:
                import pycocotools.mask as mask_util

                boxes = results["boxes"][idx : idx + box_num].tolist()
                masks = results["masks"][i][:box_num].astype(np.uint8)
                seg_res = []
                for box, mask in zip(boxes, masks):
                    rle = mask_util.encode(np.array(mask[:, :, None], dtype=np.uint8, order="F"))[0]
                    if "counts" in rle:
                        rle["counts"] = rle["counts"].decode("utf8")
                    seg_res.append(
                        {
                            "image_id": img_id,
                            "category_id": coco_clsid2catid[int(box[0])] if use_coco_category else int(box[0]),
                            "file_name": file_name,
                            "segmentation": rle,
                            "score": box[1],
                        }
                    )
                mask_results.extend(seg_res)

            idx += box_num

        if bbox_results:
            bbox_file = os.path.join(self.output_dir, "bbox.json")
            with open(bbox_file, "w") as f:
                json.dump(bbox_results, f)
            print(f"The bbox result is saved to {bbox_file}")
        if mask_results:
            mask_file = os.path.join(self.output_dir, "mask.json")
            with open(mask_file, "w") as f:
                json.dump(mask_results, f)
            print(f"The mask result is saved to {mask_file}")


class DetPretrainNet(BaseModel):
    def __init__(self, model_path: str = HERE / "ckpt"):
        """
        init model

        Args:
            model_path (str, optioanl):
        """
        super().__init__(curr_path=curr_path)

        self.model = Detector(model_path)

    def predict(self, req):
        img = self.get_image(req)
        if self.model is None:
            abort("Model is not loaded.", 404)
        pred = self.model.run(img)
        # print(pred["boxes"], img.shape)
        # h, w, c = img.shape
        predictions = []
        for b in pred["boxes"]:
            res = b[2:]
            # res[0] -= w // 2
            # res[1] -= h // 2
            # res[2] -= w // 2
            # res[3] -= h // 2
            res = map(lambda v: str(int(v)), res)
            predictions.append(
                {"label_name": self.model.pred_config.labels[int(b[0])], "score": str(b[1]), "result": ",".join(res)}
            )
        predictions.sort(key=lambda r: r["score"], reverse=True)
        return predictions


if __name__ == "__main__":
    model = Detector("/Users/haoyuying/Documents/ml_pretrain/picodet_s_416_coco_lcnet_with_postprocess")
    # image = "/Users/haoyuying/Documents/PaddleDetection/demo/000000014439.jpg"
    image = cv2.imread("/Users/haoyuying/Documents/PaddleDetection/demo/000000014439.jpg")
    output = model.run(image)

    print(output)
