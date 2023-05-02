


for seed in 0 1 2
do
python ../run_expt.py --setting WBIRDS  --save_last --save_best  --diversify  --diversity_weight 10.0 --heads 2  --root_dir /datasets/home/hbenoit/DivDis-exp --log_dir waterbirds  --seed $seed

done
