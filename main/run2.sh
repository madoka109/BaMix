#!/bin/bash
python cifar_train.py --mixup 1 --devices 0 --dataset cifar10 --imb_factor 0.01 --epochs 400 --drw_epoch 320 --loss_type STM --train_rule DRW --mixepoch 400

python cifar_train.py --mixup 1 --devices 0 --dataset cifar10 --imb_factor 0.02 --epochs 400 --drw_epoch 320 --loss_type STM --train_rule DRW --mixepoch 400

python cifar_train.py --mixup 1 --devices 0 --dataset cifar100 --imb_factor 0.01 --epochs 400 --drw_epoch 320 --loss_type STM --train_rule DRW --mixepoch 400

python cifar_train.py --mixup 1 --devices 0 --dataset cifar100 --imb_factor 0.02 --epochs 400 --drw_epoch 320 --loss_type STM --train_rule DRW --mixepoch 400
