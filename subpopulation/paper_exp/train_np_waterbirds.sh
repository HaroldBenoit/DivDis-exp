


for seed in 0 1 2
do
python ../run_expt.py --setting WBIRDS --train_from_scratch --log_dir np_waterbirds --seed $seed
python ../run_expt.py --setting WBIRDS --train_from_scratch --log_dir np_waterbirds --seed $seed
python ../run_expt.py --setting WBIRDS --train_from_scratch --log_dir np_waterbirds --seed $seed

done
