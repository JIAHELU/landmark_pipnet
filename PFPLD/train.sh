#!/usr/bin/env bash
#pfld-ori
#CUDA_VISIBLE_DEVICES='0' nohup python -u  train.py --snapshot /home/bill.zeng/dl_base/landmark/nniefacelib/PFPLD/my_models/pfld-ori -j 16 2>&1 | tee log/pfld-ori.txt &

#pfld-ori-epoch 200
CUDA_VISIBLE_DEVICES='1' nohup python -u  train.py --step "60,120" --end_epoch 160 --snapshot /home/bill.zeng/dl_base/landmark/nniefacelib/PFPLD/my_models/pfld-ori-e160 -j 8 2>&1 | tee log/pfld-ori-e160.txt &


#pfld-0.25
#CUDA_VISIBLE_DEVICES='0' nohup python -u  train.py --snapshot /home/bill.zeng/dl_base/landmark/nniefacelib/PFPLD/my_models/pfld-0.25 -j 8 2>&1 | tee log/pfld-0.25.txt &

#pfld-0.25-epoch 200
#CUDA_VISIBLE_DEVICES='1' nohup python -u  train.py --step "60,120" --end_epoch 160 --snapshot /home/bill.zeng/dl_base/landmark/nniefacelib/PFPLD/my_models/pfld-0.25-e160 -j 8 2>&1 | tee log/pfld-0.25-e160.txt &

#pfld-ori-epoch 200 -lr first 0.1
# CUDA_VISIBLE_DEVICES='0' nohup python -u  train.py --base_lr 0.1 --step "40,100,160" --end_epoch 200 --snapshot /home/bill.zeng/dl_base/landmark/nniefacelib/PFPLD/my_models/pfld-0.25-e200 -j 8 2>&1 | tee log/pfld-0.25-e200.txt &

#pfld-shufflev2x0.5
#CUDA_VISIBLE_DEVICES='0' nohup python -u  train.py --snapshot /home/bill.zeng/dl_base/landmark/nniefacelib/PFPLD/my_models/pfld-shufflev2x0.5 -j 8 2>&1 | tee log/pfld-shufflev2x0.5.txt &

#pfld-shufflev2x0.5-epoch 160
#CUDA_VISIBLE_DEVICES='2' nohup python -u  train.py --step "60,120" --end_epoch 160 --snapshot /home/bill.zeng/dl_base/landmark/nniefacelib/PFPLD/my_models/pfld-shufflev2x0.5-e160 -j 8 2>&1 | tee log/pfld-shufflev2x0.5-e160.txt &

#pfld-shufflev2x0.5-epoch-FINE 160
#CUDA_VISIBLE_DEVICES='0' nohup python -u  train.py --step "60,120" --end_epoch 160 --snapshot /home/bill.zeng/dl_base/landmark/nniefacelib/PFPLD/my_models/pfld-shufflev2x0.5-e160-fine -j 8 2>&1 | tee log/pfld-shufflev2x0.5-e160-fine.txt &

#pfld-mbv1x0.25-FINE
#CUDA_VISIBLE_DEVICES='1' nohup python -u  train.py --snapshot /home/bill.zeng/dl_base/landmark/nniefacelib/PFPLD/my_models/pfld-mbv1x0.25-fine -j 8 2>&1 | tee log/pfld-mbv1x0.25-fine.txt &

#pfld-mbv3_small
#CUDA_VISIBLE_DEVICES='1' nohup python -u  train.py --snapshot /home/bill.zeng/dl_base/landmark/nniefacelib/PFPLD/my_models/pfld-mbv3_small -j 8 2>&1 | tee log/pfld-mbv3_small.txt &


