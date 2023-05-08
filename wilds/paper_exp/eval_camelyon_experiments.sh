python ../evaluate.py /datasets/home/hbenoit/DivDis-exp/wilds/logs/div10 --dataset camelyon17 --seeds 2 3 4 --folder_format camelyon_seed{seed} --final_name camelyon
python ../evaluate.py /datasets/home/hbenoit/DivDis-exp/wilds/logs/div10 --dataset camelyon17 --seeds 2 3 4 --folder_format camelyon_np_seed{seed} --final_name camelyon_np
python ../evaluate.py /datasets/home/hbenoit/DivDis-exp/wilds/logs/ERM --dataset camelyon17 --seeds 2 3 4 --folder_format camelyon_seed{seed} --final_name ERM
python ../evaluate.py /datasets/home/hbenoit/DivDis-exp/wilds/logs/ERM --dataset camelyon17 --seeds 2 3 4 --folder_format camelyon_np_seed{seed} --final_name ERM_np

