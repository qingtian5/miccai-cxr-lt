from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
import json


@DATASETS.register_module()
class Jsonlist(BaseDataset):

    def load_data_list(self):
        assert isinstance(self.ann_file, str)

        data_list = []
        with open(self.ann_file) as f:
            data = json.load(f)
            for d in data:
                info = {'img_path': d}
                data_list.append(info)
        return data_list
