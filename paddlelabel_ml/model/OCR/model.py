import os.path as osp

from paddleocr import PaddleOCR, draw_ocr

from paddlelabel_ml.model import BaseModel
from paddlelabel_ml.util import abort
from paddlelabel_ml.util import use_gpu

curr_path = osp.abspath(osp.dirname(__file__))


class PPOCR(BaseModel):
    name = "Paddle OCR"

    def __init__(self, lang="ch", use_gpu=use_gpu):
        """
        init model
        """
        super().__init__(curr_path=curr_path)

        self.model = PaddleOCR(lang=lang, use_gpu=use_gpu)
        self.lang = lang

    def predict(self, req):
        if self.model is None:
            abort("Model is not loaded.")

        img = self.get_image(req)
        res = self.model.ocr(img=img)

        """
        help='==SUPPRESS==',
        use_gpu=False,
        use_xpu=False,
        use_npu=False,
        ir_optim=True,
        use_tensorrt=False,
        min_subgraph_size=15,
        precision='fp32',
        gpu_mem=500,
        image_dir=None,
        page_num=0,
        det_algorithm='DB',
        det_model_dir='/home/lin/.paddleocr/whl/det/ch/ch_PP-OCRv3_det_infer',
        det_limit_side_len=960,
        det_limit_type='max',
        det_box_type='quad',
        det_db_thresh=0.3,
        det_db_box_thresh=0.6,
        det_db_unclip_ratio=1.5,
        max_batch_size=10,
        use_dilation=False,
        det_db_score_mode='fast',
        det_east_score_thresh=0.8,
        det_east_cover_thresh=0.1,
        det_east_nms_thresh=0.2,
        det_sast_score_thresh=0.5,
        det_sast_nms_thresh=0.2,
        det_pse_thresh=0,
        det_pse_box_thresh=0.85,
        det_pse_min_area=16,
        det_pse_scale=1,
        scales=[8, 16, 32],
        alpha=1.0,
        beta=1.0,
        fourier_degree=5,
        rec_algorithm='SVTR_LCNet',
        rec_model_dir='/home/lin/.paddleocr/whl/rec/ch/ch_PP-OCRv3_rec_infer',
        rec_image_inverse=True,
        rec_image_shape='3, 48, 320',
        rec_batch_num=6,
        max_text_length=25,
        rec_char_dict_path='/home/lin/micromamba/envs/PaddleLabel-ML/lib/python3.10/site-packages/paddleocr/ppocr/utils/ppocr_keys_v1.txt',
        use_space_char=True,
        vis_font_path='./doc/fonts/simfang.ttf',
        drop_score=0.5,
        e2e_algorithm='PGNet',
        e2e_model_dir=None,
        e2e_limit_side_len=768,
        e2e_limit_type='max',
        e2e_pgnet_score_thresh=0.5,
        e2e_char_dict_path='./ppocr/utils/ic15_dict.txt',
        e2e_pgnet_valid_set='totaltext',
        e2e_pgnet_mode='fast',
        use_angle_cls=False,
        cls_model_dir='/home/lin/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer',
        cls_image_shape='3, 48, 192',
        label_list=['0', '180'],
        cls_batch_num=6,
        cls_thresh=0.9,
        enable_mkldnn=False,
        cpu_threads=10,
        use_pdserving=False,
        warmup=False,
        sr_model_dir=None,
        sr_image_shape='3, 32, 128',
        sr_batch_num=1,
        draw_img_save_dir='./inference_results',
        save_crop_res=False,
        crop_res_save_dir='./output',
        use_mp=False,
        total_process_num=1,
        process_id=0,
        benchmark=False,
        save_log_path='./log_output/',
        show_log=True,
        use_onnx=False,
        output='./output',
        table_max_len=488,
        table_algorithm='TableAttn',
        table_model_dir=None,
        merge_no_span_structure=True,
        table_char_dict_path=None,
        layout_model_dir=None,
        layout_dict_path=None,
        layout_score_threshold=0.5,
        layout_nms_threshold=0.5,
        kie_algorithm='LayoutXLM',
        ser_model_dir=None,
        re_model_dir=None,
        use_visual_backbone=True,
        ser_dict_path='../train_data/XFUND/class_list_xfun.txt',
        ocr_order_method=None,
        mode='structure',
        image_orientation=False,
        layout=True,
        table=True,
        ocr=True,
        recovery=False,
        use_pdf2docx_api=False,
        lang='ch',
        det=True,
        rec=True,
        type='ocr',
        ocr_version='PP-OCRv3',
        structure_version='PP-StructureV2'
        """

        results = []

        for line in res[0]:
            """
            line:
            [[[[10.0, 9.0], [347.0, 9.0], [347.0, 63.0], [10.0, 63.0]], ('韩国小馆', 0.995253324508667)]]

            result:
            p1.w|p1.h|....|pn.w|pn.h|(固定为空，表示点结束)|transcription|illegibility(0/1)|language
            """

            result = "|".join(f"{p[0]:.0f}|{p[1]:.0f}" for p in line[0]) + "||"
            result += f"{line[1][0]}|1|{self.lang}"
            results.append(
                {
                    "result": result,
                    "score": f"{line[1][1]:.4f}",
                }
            )

        return results

        # 可视化
        # from PIL import Image
        # image = Image.open(img_path).convert("RGB")
        # boxes = [line[0] for line in result]
        # txts = [line[1][0] for line in result]
        # scores = [line[1][1] for line in result]
        # im_show = draw_ocr(image, boxes, txts, scores, font_path="/path/to/PaddleOCR/doc/fonts/korean.ttf")
        # im_show = Image.fromarray(im_show)
        # im_show.save("result.jpg")

        # print("Prediction", res)
        # return res[0]["category"]
