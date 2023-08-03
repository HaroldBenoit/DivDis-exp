

for seed in 0 1 2 
do

for diversity_weight in 10 
    do
            python ../../run_expt.py --setting WBIRDS --n_epochs 100  --model resnet50 --model_list resnet50 2 robust_resnet50 2 vit_b_16 2 resnet50MocoV2 2 --majority_only --save_last --save_best --batch_size 32 --diversify --diversity_weight $diversity_weight --heads 8 --root_dir /datasets/home/hbenoit/DivDis-exp --log_dir eight_models_all_pretrained/div$diversity_weight/cc_waterbirds/seed$seed --seed $seed

    done


done

