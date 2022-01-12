srun --ntasks=1 --gres=gpu:V100:1 --cpus-per-gpu=4 --mem=32G --time=4320 \
python /scratch/kadur/src/main.py \
        --dataroot /scratch/kadur/DATA/Image \
        --n_epoch 15 \
        --n_epoch_decay 10 \
        --model BaseModel \
        --dataset PairedDataset \
        --batch_size 1 \
        --num_workers 1 \
        --load_size 270 \
        --crop_size 256 \
        --use_dis_content \
        --concat \
        --latent_dim 8 \
        --num_domains 5 \
        --dis_sn \
        --d_iter 5 \
        --name basemodel_with_sn \
        --save_freq 20000 \
        --print_freq 3000 \