python ../evaluate.py logs/ --dataset camelyon17 --seeds 2 3 4 --folder_format camelyon_seed{seed} --final_name camelyon
python ../evaluate.py logs/ --dataset camelyon17 --seeds 2 3 4 --folder_format camelyon_np_seed{seed} --final_name camelyon_np
python ../evaluate.py logs/ --dataset camelyon17 --seeds 2 3 4 --folder_format resnet_camelyon_seed{seed} --final_name resnet_camelyon
python ../evaluate.py logs/ --dataset camelyon17 --seeds 2 3 4 --folder_format resnet_np_camelyon_seed{seed} --final_name resnet_np_camelyon
