

for seed in 2 3 4
    do

    python ../run_expt.py --root_dir /datasets/home/hbenoit/D-BAT-exp/datasets/ --dataset camelyon17 --model_kwargs pretrained=False  --additional_train_transform None --algorithm ERM  --log_dir ../logs/ERM/camelyon_seed$seed --seed $seed


done


