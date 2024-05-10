export CUDA_VISIBLE_DEVICES=4

# train_path: the path of the training data
# test_path: the path of the validation data
# training_type: the type of basic
# align_score_cutoff: the threshold of the alignment score
# model_name: the name of the model

python -W ignore Train_new.py \
--train_path /home/user01/majiajian/data/polyp/SUN-SEG_10/match_data/1_data.csv \
--test_path /home/user01/majiajian/data/polyp/SUN-SEG_10/SUN-SEG_10_val \
--training_type basic \
--align_score_cutoff 0.8 \
--model_name SUN-SEG_Aug_baseline \
--augmentation True \
--lr=1e-4