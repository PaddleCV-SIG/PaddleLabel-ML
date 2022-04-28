import os
import os.path as osp

import paddlex as pdx
from paddlex import transforms as T
import paddlex

from pplabel_ml.model import BaseModel, add_model
from pplabel_ml.util import abort
from pplabel_ml.model.util import copycontent

curr_path = osp.abspath(osp.dirname(__file__))


@add_model
class PdxMobilenetv2(BaseModel):
    name = "Pdx Mobilenetv2 Classfication"

    def __init__(self, param_path=osp.join(curr_path, "ckpt", "best_model")):
        """
        init model

        Args:
            param_path (str, optional): The "best model" path, will load model from this path for inference. Defaults to osp.join(curr_path, "ckpt", "best_model").
        """
        super().__init__(curr_path=curr_path)
        self.param_path = param_path
        if osp.exists(param_path):
            self.model = pdx.load_model(param_path)
        else:
            self.model = None

        self.pretrain_weights_url = "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_pretrained.pdparams"
        self.pretrain_weights_path = osp.join(
            self.ckpt_path, "pretrain", "MobileNetV2_pretrained.pdparams"
        )
        if not osp.exists(self.pretrain_weights_path):
            paddlex.utils.download.download(
                self.pretrain_weights_url, osp.dirname(self.pretrain_weights_path)
            )
        self.training = False

        print("output_path", self.output_path)

    def predict(self, req):
        img = self.get_image(req)
        if self.model is None:
            abort("Model is not loaded.")
        res = self.model.predict(img)
        return res[0]["category"]

    def train(self, data_dir):
        self.pretrain_check(data_dir)

        train_transforms = T.Compose(
            [T.RandomCrop(crop_size=224), T.RandomHorizontalFlip(), T.Normalize()]
        )
        eval_transforms = T.Compose(
            [T.ResizeByShort(short_size=256), T.CenterCrop(crop_size=224), T.Normalize()]
        )

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

        # print("save path", self.param_path, osp.basename(self.param_path))
        model.train(
            num_epochs=15,
            train_dataset=train_dataset,
            train_batch_size=8,  # 200m/sample
            eval_dataset=eval_dataset,
            lr_decay_epochs=[4, 6, 8],
            save_interval_epochs=1,
            learning_rate=0.025,
            save_dir=self.output_path,
            pretrain_weights=self.pretrain_weights_path,
            use_vdl=True,
        )
        copycontent(osp.join(self.output_path, "best_model"), osp.join(self.ckpt_path, "best_model"))
        self.model = pdx.load_model(self.param_path)
