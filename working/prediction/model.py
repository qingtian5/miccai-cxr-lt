import os
import cv2
import torch
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from mmpretrain.apis import init_model, ImageClassificationInferencer
import argparse


class model:
    def __init__(self, checkpoint, config_path):
        self.checkpoint = checkpoint
        self.config_path = config_path
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

        with torch.no_grad():
            predict = self.inference(input_image_path, batch_size=1)

        return predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--checkpoint', type=str, default="epoch_12.pth", help='check point path')
    parser.add_argument('--config', type=str, default="mae_vit-large-mae-ft-asymmetricloss-unbalanced-data.py", help='config_path')
    args = parser.parse_args()
    checkpoint = args.checkpoint
    config = args.config

    valid_data_path = "/mnt/pfs-guan-ssai/cv/panxuhao/misc/playground/miccai24-cxr-lt/task1_development_starting_kit/development.csv"
    classes_path = "/mnt/pfs-guan-ssai/cv/panxuhao/misc/playground/miccai24-cxr-lt/task1_development_starting_kit/EVAL_CLASSES.txt"
    valid_data = pd.read_csv(valid_data_path)
    with open(classes_path, "r") as f:
        classes = f.read().splitlines()

    out_data = valid_data[["dicom_id", "fpath"]].copy()

    # test the model
    model = model(checkpoint, config)
    temp = config.split(".")[0]
    epoch_temp = checkpoint.split(".")[0]
    model.load(f"/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/train/work_dirs/{temp}")

    input_image_path = []
    for i in out_data["fpath"]:
        input_image_path.append(os.path.join("/mnt/pfs-guan-ssai/cv/panxuhao/misc/playground/physionet.org/files/mimic-cxr-jpg/2.0.0", i))

    outputs = model.predict(input_image_path)
    predictions = [out["pred_scores"] for out in outputs]
    predictions_df = pd.DataFrame(predictions, columns=classes)
    id_df = out_data["dicom_id"].copy()
    out_df = pd.concat([id_df, predictions_df], axis=1)
    des_dir = f"/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/prediction/submission/{temp}/{epoch_temp}"
    os.makedirs(des_dir, exist_ok=True)
    out_df.to_csv(f'{des_dir}/development_task1_sample_submission.csv', index=False)


