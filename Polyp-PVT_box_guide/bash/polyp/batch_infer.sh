export CUDA_VISIBLE_DEVICES=7

python -W ignore batch_infer.py \
--images_root "/home/user01/majiajian/data/polyp/SUN-SEG_10/multiple_controlnet_inpaint/images" \
--coarse_mask_root "/home/user01/majiajian/data/polyp/SUN-SEG_10/multiple_controlnet_inpaint/initial_masks" \
--output_path "/home/user01/majiajian/data/polyp/SUN-SEG_10/multiple_controlnet_inpaint" \
--condition_mask_root "/home/user01/majiajian/data/polyp/SUN-SEG_10/multiple_controlnet_inpaint/conditions" \
--resolution 512 \
--iters 5 \
--testsize 352 \
--pth_path "/home/user01/majiajian/code/segmentation/Polyp-PVT_box_guide/model_pth/SUNSEG_guided_2024-05-01-21-38-56/6PolypPVT-best.pth" \
--pth_path_original "/home/user01/majiajian/code/segmentation/Polyp-PVT/model_pth/SUN-SEG_baseline_2024-04-30-16-13-35/14PolypPVT-best.pth" 