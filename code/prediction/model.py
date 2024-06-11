import os
import cv2
import torch
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from mmpretrain.apis import init_model, ImageClassificationInferencer


class model:
    def __init__(self):
        self.checkpoint = "epoch_36.pth"
        self.config_path = "mae_vit-large-mae-ft-unbalanced-data.py"
        # The model is evaluated using CPU, please do not change to GPU to avoid error reporting.
        self.device = torch.device("cpu")

    def load(self, dir_path):
        checkpoint_path = os.path.join(dir_path, self.checkpoint)
        config_path = os.path.join(dir_path, self.config_path)
        self.model = init_model(config_path,checkpoint=checkpoint_path,device="cpu")
        self.model.eval()
        self.inference = ImageClassificationInferencer(self.model)

    def predict(self, input_image_path):
        pred = []
        for idx, i in enumerate(input_image_path):
            input_image = cv2.imread(i)
            with torch.no_grad():
                predict = self.inference([input_image], batch_size=1)
                pred.append(predict[0]["pred_scores"])
            if idx % 100 == 0:
                print(f"Processed {idx} images")
        return pred

if __name__ == "__main__":
    valid_data_path = "/mnt/pfs-guan-ssai/cv/panxuhao/misc/playground/miccai24-cxr-lt/task1_development_starting_kit/development.csv"
    classes_path = "/mnt/pfs-guan-ssai/cv/panxuhao/misc/playground/miccai24-cxr-lt/task1_development_starting_kit/EVAL_CLASSES.txt"
    valid_data = pd.read_csv(valid_data_path)
    with open(classes_path, "r") as f:
        classes = f.read().splitlines()

    out_data = valid_data[["dicom_id", "fpath"]].copy()

    # test the model
    model = model()
    model.load("/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/train/work_dirs/mae_vit-large-mae-ft-unbalanced-data")

    input_image_path = []
    for i in out_data["fpath"]:
        input_image_path.append(os.path.join("/mnt/pfs-guan-ssai/cv/panxuhao/misc/playground/physionet.org/files/mimic-cxr-jpg/2.0.0", i))

    outputs = model.predict(input_image_path)
    predictions_df = pd.DataFrame(outputs, columns=classes)
    id_df = out_data["dicom_id"].copy()
    out_df = pd.concat([id_df, predictions_df], axis=1)
    des_dir = "/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/prediction/submission/mae_vit-large-mae-ft-unbalanced-data/epoch_36_real"
    os.makedirs(des_dir, exist_ok=True)
    out_df.to_csv(f'{des_dir}/development_task1_sample_submission.csv', index=False)


