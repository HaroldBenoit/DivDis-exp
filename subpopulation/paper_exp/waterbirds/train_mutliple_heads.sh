

for diversity_weight in 10 1 0.1
do
for heads in 3 4 5
    do
        for seed in 0 1 2
        do
            python ../../run_expt.py --setting WBIRDS --n_epochs 100  --majority_only --train_from_scratch --save_last --save_best --batch_size 128 --diversify --diversity_weight $diversity_weight --heads $heads --root_dir /datasets/home/hbenoit/DivDis-exp --log_dir heads$heads/div$diversity_weight/np_cc_waterbirds/seed$seed  --seed $seed
        done
    done

done
