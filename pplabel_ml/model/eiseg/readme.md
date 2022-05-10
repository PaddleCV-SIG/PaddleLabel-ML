# 安装依赖
```shell script
pip install -r requirements.txt
```

# 使用接口的方式：
```shell script
cd eiseg_static

python api.py
```
使用的例子可以看api.py中是如何调用调用的。

# API接口说明

**Predictor初始化**
=====================

```python
def __init__():
```
* 无参数


**member functions**
=====================

run
------------------

```
def run(self, 
        image: np.array, 
        clicker_list: List, 
        pred_thr: float = 0.49, 
        pred_mask: np.array = None):
```

用于设定选定的模型

* Args:
    * image(np.array):
         原始图像，rgb通道。
    * clicker_list(List):
         传入clicker_list的格式为[(x,y,is_positive),(x,y,is_positive),...].其中，x为点击横坐标，y为纵坐标，
         is_positive用于区分是否是正点。
         
* Return：
    * output(np.array)：输出的mask结果
    * pred_probs(np.array)：用于下一次pred_mask输入

# 界面应用
若想使用界面体验可以使用:

```shell script
python demo.python
```