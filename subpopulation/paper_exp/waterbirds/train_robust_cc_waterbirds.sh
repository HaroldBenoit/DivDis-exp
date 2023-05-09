

for diversity_weight in 10
    do
        for seed in 0 1 2
        do
            python ../../run_expt.py --setting WBIRDS --model robust_resnet50 --majority_only --save_last --save_best --diversify --diversity_weight $diversity_weight --heads 2 --root_dir /datasets/home/hbenoit/DivDis-exp --log_dir robust/div$diversity_weight/cc_waterbirds/seed$seed  --seed $seed
        done
    done


