# export CUDA_VISIBLE_DEVICES=7
export MODEL_DIR="/home/user01/majiajian/code/diffusion/diffusers/output_model/sd_inpaint/inpaint_1e5"
export VALIDATION_LIST="/home/user01/majiajian/code/diffusion/diffusers/examples/controlnet/concept_lists/polyp/0430/validation_list_0430.json"
export OUTPUT_DIR="/home/user01/majiajian/code/diffusion/diffusers/output_model/multicontrolnet"
export BOUNDARY_CONTROL_MODEL_DIR="/home/user01/majiajian/pretrain_model/control_v11p_sd15_seg"
export SURFACE_CONTROL_MODEL_DIR="/home/user01/majiajian/pretrain_model/control_v11e_sd15_shuffle"
# export HF_HUB_OFFLINE=1

# model dir is the path of the finetuned stable diffusion backbone model
# validation_list is the path of the validation list
# boundary_controlnet is the path of the boundary controlnet model (seg)
# surface_controlnet is the path of the surface controlnet model (shuffle)
# output_dir is the path of the output directory
# instance_data_root is the path of the polyp images
# instance_mask_root is the path of the mask of the polyp
# instance_prompts is the prompt of the polyp
# subfolders is the flag of whether the instance_data_root is the path of the subfolders, suitible for SUN-SEG_10


accelerate launch script/polyp/train_multicontrolnet_inpaint_0430.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --instance_data_root="/home/user01/majiajian/data/polyp/SUN-SEG_10/TrainDataset_10/images" \
 --instance_mask_root="/home/user01/majiajian/data/polyp/SUN-SEG_10/TrainDataset_10/masks" \
 --instance_prompts="an endoscopic image of a polyp" \
 --subfolders \
 --boundary_controlnet=$BOUNDARY_CONTROL_MODEL_DIR \
 --surface_controlnet=$SURFACE_CONTROL_MODEL_DIR \
 --resolution=512 \
 --learning_rate=5e-5 \
 --validation_list=$VALIDATION_LIST \
 --train_batch_size=4 \
 --tracker_project_name="multicontrolnet" \
 --enable_xformers_memory_efficient_attention \
 --max_train_steps=20000 \
 --checkpointing_steps=4000 \
 --validation_steps=1000 \
 --mixed_precision="fp16" \
 --seed=42 



#  --controlnet_model_name_or_path=$CONTROL_MODEL_DIR \
#  --resume_from_checkpoint="latest" \