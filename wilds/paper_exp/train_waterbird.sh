


for seed in 0 1 2
    do
    
    python ../run_expt.py --root_dir /datasets/home/hbenoit/D-BAT-exp/datasets/ --dataset waterbirds --unlabeled_split val --algorithm DivDis --additional_train_transform None --divdis_diversity_weight 1 --log_dir ../logs/waterbirds_seed$seed --seed $seed

    done