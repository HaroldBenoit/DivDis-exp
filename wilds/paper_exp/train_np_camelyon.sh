
## possibly increase epoch 
for seed in 3 4
    do

    for diversity_weight in 1 10
    do

        python ../run_expt.py --root_dir /datasets/home/hbenoit/D-BAT-exp/datasets/ --dataset camelyon17 --model_kwargs pretrained=False --unlabeled_split test_unlabeled --additional_train_transform None --algorithm DivDis --divdis_diversity_weight $diversity_weight --log_dir ../logs/div$diversity_weight/camelyon_np_seed$seed --seed $seed

    done
    done

