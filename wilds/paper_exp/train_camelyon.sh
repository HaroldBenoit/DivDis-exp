


for seed in 3 4
    do
    
    python ../run_expt.py --root_dir /datasets/home/hbenoit/D-BAT-exp/datasets/ --dataset camelyon17 --model_kwargs pretrained=True --unlabeled_split test_unlabeled --additional_train_transform None --algorithm DivDis --divdis_diversity_weight 0.01 --log_dir ../logs/camelyon_seed$seed --seed $seed

    done