

for diversity_weight in 10 1 0.1
    do
        for seed in 0 1 2
        do
            python ../../run_expt.py --setting WBIRDS --n_epochs 100  --model vit_b_16 --majority_only --save_last --save_best --diversify --batch_size 128 --diversity_weight $diversity_weight --heads 2 --root_dir /datasets/home/hbenoit/DivDis-exp --log_dir vit_b_16/div$diversity_weight/cc_waterbirds/seed$seed  --seed $seed
        done
    done


