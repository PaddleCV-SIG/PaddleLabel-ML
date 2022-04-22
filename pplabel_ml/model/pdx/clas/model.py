import os
import os.path as osp

import paddlex as pdx
from paddlex import transforms as T

from pplabel_ml.model import BaseModel, add_model


curr_path = osp.abspath(osp.dirname(__file__))


@add_model
class PdxMobilenetv2(BaseModel):
    name = "Pdx Mobilenetv2 Classfication"

    def __init__(self, param_path=osp.join(curr_path, "ckpt", "best_model")):
        self.param_path = param_path
        self.model = pdx.load_model(param_path)
        self.training = False

    def predict(self, img):
        img = self.decodeb64(img)
        res = self.model.predict(img)
        return res[0]["category"]

    def train(self, data_dir):

        train_transforms = T.Compose(
            [T.RandomCrop(crop_size=224), T.RandomHorizontalFlip(), T.Normalize()])

        eval_transforms = T.Compose([
            T.ResizeByShort(short_size=256), T.CenterCrop(crop_size=224), T.Normalize()
        ])

        train_dataset = pdx.datasets.ImageNet(
            data_dir=data_dir,
            num_workers=0,
            file_list=osp.join(data_dir, "train_list.txt"),
            label_list=osp.join(data_dir, "labels.txt"),
            transforms=train_transforms,
            shuffle=True,
        )
        eval_dataset = pdx.datasets.ImageNet(
            data_dir=data_dir,
            num_workers=0,
            file_list=osp.join(data_dir, "val_list.txt"),
            label_list=osp.join(data_dir, "labels.txt"),
            transforms=eval_transforms,
        )

        num_classes = len(train_dataset.labels)

        model = pdx.cls.MobileNetV2(num_classes=num_classes)

        model.train(
            num_epochs=15,
            train_dataset=train_dataset,
            train_batch_size=8, # 200m/sample
            eval_dataset=eval_dataset,
            lr_decay_epochs=[4, 6, 8],
            save_interval_epochs=1,
            learning_rate=0.025,
            save_dir=osp.basename(self.param_path),
            use_vdl=True,
        )

# https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_pretrained.pdparams