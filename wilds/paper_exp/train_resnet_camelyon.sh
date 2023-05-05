


for seed in 2 3 4
    do
    
    python ../run_expt.py --root_dir /datasets/home/hbenoit/D-BAT-exp/datasets/ --resume --dataset camelyon17 --model resnet50 --model_kwargs pretrained=True --unlabeled_split test_unlabeled --additional_train_transform None --algorithm DivDis --divdis_diversity_weight 0.01 --log_dir ../logs/resnet_camelyon_seed$seed --seed $seed

    done