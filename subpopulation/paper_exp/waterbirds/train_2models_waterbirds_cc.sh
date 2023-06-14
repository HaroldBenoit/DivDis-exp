

for seed in 0 1 2
do

for diversity_weight in 10 
    do
            python ../../run_expt.py --setting WBIRDS --model resnet50_resnet50_np --n_epochs 100 --majority_only --save_last --save_best --batch_size 32 --diversify --diversity_weight $diversity_weight --heads 2 --root_dir /datasets/home/hbenoit/DivDis-exp --log_dir resnet50_resnet50_np/div$diversity_weight/cc_waterbirds/seed$seed --seed $seed

            python ../../run_expt.py --setting WBIRDS --model vit_b_16_resnet50_np --n_epochs 100  --majority_only --save_last --save_best --batch_size 32 --diversify --diversity_weight $diversity_weight --heads 2 --root_dir /datasets/home/hbenoit/DivDis-exp --log_dir vit_b_16_resnet50_np/div$diversity_weight/cc_waterbirds/seed$seed --seed $seed

    done


done