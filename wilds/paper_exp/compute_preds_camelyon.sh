python results.py --seeds 2 --model densenet121 --folder_format camelyon_seed{seed} 
python results.py --seeds 2 --model densenet121 --folder_format camelyon_np_seed{seed} 
python results.py --seeds 2 --model resnet50    --folder_format resnet_camelyon_seed{seed} 
python results.py --seeds 2 --model resnet50    --folder_format resnet_np_camelyon_seed{seed}



python results.py logs/ --seeds 3 4 --model densenet121 --folder_format camelyon_seed{seed} 
python results.py logs/ --seeds 3 4 --model densenet121 --folder_format camelyon_np_seed{seed} 
python results.py logs/ --seeds 3 4 --model resnet50    --folder_format resnet_camelyon_seed{seed} 
python results.py logs/ --seeds 3 4 --model resnet50    --folder_format resnet_np_camelyon_seed{seed} 
