#!/bin/bash

# prepare environment
cd /mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/mmpretrain
pip install -U openmim && mim install -e .

# pretrain MAE
bash ./tools/dist_train.sh /mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/train/mae_vit-large-p16_8xb512-1600e_in37w.py 8

# finetune CXR-LT dataset with FocalLoss
bash ./tools/dist_train.sh /mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/train/mae_vit-large-mae-ft-unbalanced-data.py 8

# finetune CXR-LT dataset with asymmetricloss
bash ./tools/dist_train.sh /mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/train/mae_vit-large-mae-ft-asymmetricloss-unbalanced-data.py 8

