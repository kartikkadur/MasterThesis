srun --ntasks=1 --gres=gpu:V100:1 --cpus-per-gpu=4 --mem=32G --time=4320 \
python /scratch/kadur/src/main.py \
        --dataroot /scratch/kadur/DATA/SimulatedData \
        --n_epoch 2000 \
        --n_epoch_decay 1500 \
        --model BaseModel \
        --dataset PairedDataset \
        --batch_size 1 \
        --num_workers 1 \
        --load_size 270 \
        --crop_size 256 \
        --use_dis_content \
        --concat \
        --latent_dim 8 \
        --num_domains 4 \
        --d_iter 3 \
        --name basemodel_simulated_data \
        --save_freq 10000 \
        --print_freq 3000 \
        --display_freq 3000 \