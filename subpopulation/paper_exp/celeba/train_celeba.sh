

for seed in 0 1 2
do
    python ../../run_expt.py --setting CELEBA_2 --majority_only --save_last --save_best --diversify --diversity_weight 10 --heads 2 --root_dir /datasets/home/hbenoit/DivDis-exp/data/celeba --log_dir celeba_2_cc/seed$seed  --seed $seed
done

