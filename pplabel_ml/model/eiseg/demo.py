# import argparse
# import tkinter as tk
# from inference import utils
# import paddle
# from util import exp
# from interactive_demo.app import InteractiveDemoApp
# from models.is_hrnet_model import HRNetModel
# import paddle.inference as paddle_infer

# def main():
#     args, cfg = parse_args()
#     config = paddle_infer.Config('static_hrnet18s_cocolvis_model.pdmodel',
#                                  'static_hrnet18s_cocolvis_model.pdiparams')
#     config.enable_mkldnn()
#     config.enable_mkldnn_bfloat16()
#     config.switch_ir_optim(True)

#     config.set_cpu_math_library_num_threads(10)
#     #config.set_mkldnn_cache_capacity(1)
#     print(config.mkldnn_enabled())
#     model = paddle_infer.create_predictor(config)
#     print('Loaded trained params of model successfully')
#     #model.eval()
#     root = tk.Tk()
#     root.minsize(960, 480)
#     app = InteractiveDemoApp(root, args, model)
#     root.deiconify()
#     app.mainloop()


# def parse_args():
#     parser = argparse.ArgumentParser()

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


# if __name__ == '__main__':
#     main()
