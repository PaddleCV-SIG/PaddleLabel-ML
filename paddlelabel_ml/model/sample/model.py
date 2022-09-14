import os.path as osp
import time
from pathlib import Path

from paddlelabel_ml.model import BaseModel
from paddlelabel_ml.util import abort


class SampleModel(BaseModel):
    def __init__(self, model_path: str = None, param_path: str = None):
        """
        init model

        Args:
            model_path (str, optioanl):
            param_path (str, optional):
        """
        super().__init__(curr_path=Path(__file__).parent.absolute())

    def predict(self, req):
        time.sleep(10)
        return "finished"


print("Sample called")
