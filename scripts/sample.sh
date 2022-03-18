#!/bin/bash
python sample.py --dataroot $1 --model AdaINModel --latent_dim 8 --num_domains 4 --targets cloud fog rain sun --mode test --out_fmt image --resume ./checkpoint/model.ckpt --reparam --concat
