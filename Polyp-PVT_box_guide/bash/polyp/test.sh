export CUDA_VISIBLE_DEVICES=2

python -W ignore Test.py \
--datasets 'SUN-SEG_10_test+Kvasir-SEG+CVC-ClinicDB+CVC-300+ETIS-LaribPolypDB' \
--train_dataset_name polyp \
--data_path /home/user01/majiajian/data/polyp \
--save_grid \
--save_res \
--pth_path "/home/user01/majiajian/code/segmentation/Polyp-PVT_box_guide/model_pth/SUNSEG_guided_2024-05-01-21-38-56/6PolypPVT-best.pth"
