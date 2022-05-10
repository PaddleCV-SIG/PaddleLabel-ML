# import argparse
# #import tkinter as tk
# #from inference import utils
# import paddle
# from util import exp
# #from interactive_demo.app import InteractiveDemoApp
# from models.is_hrnet_model import HRNetModel
# import os
# from paddleseg.utils import logger
# import yaml


# def main():
#     #args, cfg = parse_args()
#     #小模型部分
#     model = HRNetModel(width=18, ocr_width=48, small=True, with_aux_output=True, use_rgb_conv=False,
#                        use_leaky_relu=True, use_disks=True, with_prev_mask=True, norm_radius=5, cpu_dist_maps=False)
#     #通用
#     #para_state_dict = paddle.load('weights/hrnet18s_ocr48_cocolvis.pdparams')
#     #人像
#     para_state_dict = paddle.load('/Users/haoyuying/Downloads/hrnet18s_ocr48_cocolvis.pdparams')

   
#     #大模型部分，有mask
#     #model = HRNetModel(width=18, ocr_width=64, small=False, with_aux_output=True, use_leaky_relu=True,use_rgb_conv=False,
#     #                   use_disks=True, norm_radius=5, with_prev_mask=True,cpu_dist_maps=False)


#     model.set_dict(para_state_dict)
#     print('Loaded trained params of model successfully')
#     model.eval()
#     new_net = paddle.jit.to_static(
#         model,
#         input_spec=[
#             paddle.static.InputSpec(
#                 shape=[None, 3, None, None], dtype='float32'),
#             paddle.static.InputSpec(
#                 shape=[None, 3, None, None], dtype='float32')
#         ])

#     paddle.jit.save(new_net, 'static_hrnet18s_cocolvis_model')

#     yml_file = os.path.join('static_model', 'hrnet18s_cocolvis.yaml')
#     with open(yml_file, 'w') as file:
#         # transforms = cfg.export_config.get('transforms', [{
#         #     'type': 'Normalize'
#         # }])
#         data = {
#             'Deploy': {
#                 # 'transforms': transforms,
#                 'model': 'hrnet18s_cocolvis_model.pdmodel',
#                 'params': 'hrnet18s_cocolvis_model.pdiparams'
#             }
#         }
#         yaml.dump(data, file)

#     logger.info(f'Model is saved in model_dir.')
#     # root = tk.Tk()
#     # root.minsize(960, 480)
#     # app = InteractiveDemoApp(root, args, model)
#     # root.deiconify()
#     # app.mainloop()


# def parse_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--checkpoint', type=str, required=True,
#                         help='The path to the checkpoint. '
#                              'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
#                              'or an absolute path. The file extension can be omitted.')

#     parser.add_argument('--gpu', type=int, default=0,
#                         help='Id of GPU to use.')

#     parser.add_argument('--cpu', action='store_true', default=False,
#                         help='Use only CPU for inference.')

#     parser.add_argument('--limit-longest-size', type=int, default=800,
#                         help='If the largest side of an image exceeds this value, '
#                              'it is resized so that its largest side is equal to this value.')

#     parser.add_argument('--cfg', type=str, default="config.yml",
#                         help='The path to the config file.')

#     args = parser.parse_args()
#     cfg = exp.load_config_file(args.cfg, return_edict=True)

#     return args, cfg


# if __name__ == "__main__":
#    main()

