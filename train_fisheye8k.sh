#!/bin/bash

mim train mmdet \
  $(pwd)/projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py \
  --work-dir $(pwd)/exp1 \
  --gpus 0
