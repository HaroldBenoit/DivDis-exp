

for diversity_weight in 10
    do
        for seed in 0 1 2
        do
            python ../../run_expt.py --setting WBIRDS --majority_only --train_from_scratch --save_last --save_best --diversify --diversity_weight $diversity_weight --heads 2 --root_dir /datasets/home/hbenoit/DivDis-exp --log_dir div$diversity_weight/np_cc_waterbirds/seed$seed  --seed $seed
        done
    done
