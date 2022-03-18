#!/bin/bash
python train.py --dataroot $1 --model AdaINModel --dataset PairedDataset --batch_size 1 --num_workers 1 --use_dis_content --num_domains 4 --concat --reparam
