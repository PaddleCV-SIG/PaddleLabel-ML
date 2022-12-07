# PaddleLabel-ML

[PaddleLabel](https://github.com/PaddleCV-SIG/PaddleLabel)机器学习辅助标注后端。

PaddleLabel-ML 中的模型分为两类：自动推理模型和交互式模型。所有模型在分发时都包含一套默认权重，部分模型支持指定非默认权重。目前模型包括

交互式模型：

- [EISeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/EISeg)

自动推理模型：

- PicoDet
- PPLcNet

## 安装说明

### 通过 PIP 安装

```shell
pip install paddlelabel-ml
```

### 安装最新开发版

PaddleLabel 开发团队会不定期从最新的 develop 分支中使用 Github Action 构建开发版安装包。开发版较 pypi 版本经过测试较少，可能存在更多的 bug。开发版中会包含最新的功能和修复。

安装开发版的步骤为

1. 访问 PaddleLabel-ML 构建 Github Action [页面](https://github.com/PaddleCV-SIG/PaddleLabel-ML/actions/workflows/pypi.yml)，点击进入最上方（最新的一次）Action 运行。

![img](https://user-images.githubusercontent.com/29757093/206052923-d15bb06f-5ffb-4e3f-8946-f28f0d1dbaf7.png)

2. 点击下载构建出的安装包

![img](https://user-images.githubusercontent.com/29757093/206053029-21d09105-a80e-45c0-9d26-0ad622e51188.png)

3. 解压下载的安装包，其中应包含两个文件。之后使用 pip 安装其中 whl 结尾的文件，如 paddlelabel_ml-0.5.0-py3-none-any.whl。不同版本的版本号会有不同

```shell
# 注意修改命令中文件名部分
pip install paddlelabel_ml-[版本号]-py3-none-any.whl
```

### 通过源码安装

首先将代码克隆到本地：

```shell
git clone https://github.com/PaddleCV-SIG/PaddleLabel-ML
```

安装：

```shell
cd PaddleLabel-ML
python setup.py install
```

## 启动

完成上述的安装操作后，可以直接在终端使用如下指令启动 PaddleLabel 的机器学习端。

```shell
paddlelabel_ml  # 启动ml后端
```

## \*EISeg模型下载

| 模型类型     | 适用场景             | 模型结构            | 模型下载地址                                                                                                                       |
| ------------ | -------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 高精度模型   | 通用场景的图像标注   | HRNet18_OCR64       | [static_hrnet18_ocr64_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr64_cocolvis.zip)                       |
| 轻量化模型   | 通用场景的图像标注   | HRNet18s_OCR48      | [static_hrnet18s_ocr48_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_cocolvis.zip)                     |
| 高精度模型   | 通用图像标注场景     | EdgeFlow            | [static_edgeflow_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_edgeflow_cocolvis.zip)                                 |
| 高精度模型   | 人像标注场景         | HRNet18_OCR64       | [static_hrnet18_ocr64_human](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr64_human.zip)                             |
| 轻量化模型   | 人像标注场景         | HRNet18s_OCR48      | [static_hrnet18s_ocr48_human](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_human.zip)                           |
| 轻量化模型   | 遥感建筑物标注场景   | HRNet18s_OCR48      | [static_hrnet18_ocr48_rsbuilding_instance](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr48_rsbuilding_instance.zip) |
| 高精度模型\* | x 光胸腔标注场景     | Resnet50_Deeplabv3+ | [static_resnet50_deeplab_chest_xray](https://paddleseg.bj.bcebos.com/eiseg/0.5/static_resnet50_deeplab_chest_xray.zip)             |
| 轻量化模型   | 医疗肝脏标注场景     | HRNet18s_OCR48      | [static_hrnet18s_ocr48_lits](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_lits.zip)                             |
| 轻量化模型\* | MRI 椎骨图像标注场景 | HRNet18s_OCR48      | [static_hrnet18s_ocr48_MRSpineSeg](https://paddleseg.bj.bcebos.com/eiseg/0.5/static_hrnet18s_ocr48_MRSpineSeg.zip)                 |
| 轻量化模型\* | 质检铝板瑕疵标注场景 | HRNet18s_OCR48      | [static_hrnet18s_ocr48_aluminium](https://paddleseg.bj.bcebos.com/eiseg/0.5/static_hrnet18s_ocr48_aluminium.zip)                   |
