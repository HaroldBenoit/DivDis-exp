

for seed in 2 3 4
    do
        for diversity_weight in 100

            do
    
    python ../run_expt.py --root_dir /datasets/home/hbenoit/D-BAT-exp/datasets/ --dataset camelyon17 --model_kwargs pretrained=True --unlabeled_split test_unlabeled --additional_train_transform None --algorithm DivDis --divdis_diversity_weight $diversity_weight --log_dir ../logs/div$diversity_weight/camelyon_seed$seed --seed $seed

    done
done


