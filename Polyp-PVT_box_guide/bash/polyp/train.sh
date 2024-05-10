export CUDA_VISIBLE_DEVICES=1

python -W ignore Train.py \
--switch_ratio 0.0 \
--model_name SUNSEG_guided \
--train_path /home/user01/majiajian/data/polyp/SUN-SEG_10/TrainDataset_10 \
--test_path /home/user01/majiajian/data/polyp/SUN-SEG_10/SUN-SEG_10_val \
--augmentation True 