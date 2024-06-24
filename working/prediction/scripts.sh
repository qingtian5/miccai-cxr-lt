#!/bin/bash

python model.py --checkpoint=epoch_3.pth --config=swinv2-large-ft-focalloss-unbalanced-data_alltrain_data_e3.py

python model.py --checkpoint=epoch_15.pth --config=mae_vit-large-mae-e1600-ft-unbalanced-data_e15.py
