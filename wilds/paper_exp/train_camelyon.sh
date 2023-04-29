


for seed in 0 1 2
    do
    
    python ../run_expt.py --root_dir /datasets/home/hbenoit/D-BAT-exp/datasets/ --dataset camelyon17 --unlabeled_split test_unlabeled --algorithm DivDis --divdis_diversity_weight 0.01 --log_dir ../logs/camelyon_seed$seed --seed $seed

    done