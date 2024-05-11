# Ai-Security-Final-Project
The final project for DDA 6113 AI Security

Student Name: Jiajian Ma

Student ID: 222041049

## Overview
This project is the implementation of my final project **"Enhancing Generalization and Reducing Bias in Polyp
Segmentation with Diffusion-Based Inpainting Models"**

The overview of our data augmentation pipeline is shown below:
![image](https://github.com/497662892/Ai-Security-Final-Project/blob/main/imgs/pipeline.png)
which includes 5 different components: Training Inpainting Model, Generating Inpainting Images, Pseudo-mask Refinement, Suitible cases selection, and Training Segmentation Model.

The high-level idea of our work is to leverage **inpainting** based augmentation method as a **do-operator** for causal and non-causal factor, which can help to break their association and enhance the generalization and reducing bias simutaneously.
![image](https://github.com/497662892/Ai-Security-Final-Project/blob/main/imgs/casual.PNG)

The framework of our proposed inpainting method is shown below:
![image](https://github.com/497662892/Ai-Security-Final-Project/blob/main/imgs/inpaint_model.PNG)
![image](https://github.com/497662892/Ai-Security-Final-Project/blob/main/imgs/refinement.PNG)

## Dataset & Model
The dataset and model we use in this project are available at baidupan:

链接：https://pan.baidu.com/s/1ku1pImibv5yhpwzebvT1WQ?pwd=czzk 
提取码：czzk 

### Dataset
In this link, the "data.zip" file contains all of the dataset we used and synthetic images that we generated in this project.

"data/SUN-SEG_10/TrainDataset_10" contains the training set we used for baseline segmentation model training and inpainting model training.

"data/SUN-SEG_10/SUN-SEG_10_val" contains the validation set we used for baseline segmentation model training and inpainting model training.

"data/SUN-SEG_10_test" contains the test set we used for internal segmentation model testing set.

"data/CVC-300", "data/CVC-ClinicDB", "data/Kvasir-SEG", "data/ETIS-LaribPolypDB" are external segmentation model testing sets.

"data/negatives" contains the negative background images we used for inpainting.

"data/SUN-SEG_10/multiple_controlnet_inpaint" contains all of the synthetic images we generated in this project, although we 
generate 5 folds, only use the 1st fold for further training:

- "data/SUN-SEG_10/multiple_controlnet_inpaint/images" contain the synthetic images
- "data/SUN-SEG_10/multiple_controlnet_inpaint/refined_masks" contain the refine pseudo-mask of the synthetic images
- "data/SUN-SEG_10/multiple_controlnet_inpaint/initial_masks" contain the inpainted region of the synthetic images, which is used for pseudo-mask refinement.
- "data/SUN-SEG_10/match_data/1_data.csv" contains the match data of the synthetic images and the original images, which is used for further training.

### Model
In this link, the folder "model" contains all of the models we used in this project.

#### "model/diffusion_and_controlnet" subfolder
"model/diffusion_and_controlnet" subfolder contains the inpainting model and controlnet model we used in this project.

"model/diffusion_and_controlnet/pretrained" contrains the pretrained inpainting model and controlnet modules, which is used in finetuning the inpainting model. (no need to download if you don't want to finetune the inpainting model)

"model/diffusion_and_controlnet/finetuned" contrains the finetuned inpainting model and controlnet modules, which is used in generating synthetic images.

- "model/diffusion_and_controlnet/finetuned/inpaint_1e5" is the finetuned stable diffusion backbone model.
- "model/diffusion_and_controlnet/finetuned/boundary_controlnet" is the finetuned boundary controlnet model.
- "model/diffusion_and_controlnet/finetuned/surface_controlnet" is the finetuned surface controlnet model.

#### "model/refinement_network" and "model/segmentation" subfolder
"model/refinement_network" subfolder contains the pseudo-mask refinement network we used in this project.

Before running any code in the "Polyp-PVT" or "Polyp-PVT_box_guide" folder, please download the "pretrained_pth" to the corresponding folder.

"model/segmentation/model_pth" subfolder contains the baseline segmentation model and augmentation segmentation model we used in this project. can be used to test and verified our results.

"model/refinement_network/model_pth" subfolder contains the pseudo-mask refinement network we used in this project. can be used to refine the pseudo-mask.

## Environment
please run the following code for environment setup:
```bash
git clone https://github.com/497662892/Ai-Security-Final-Project.git
cd Ai-Security-Final-Project
pip install -r requirements.txt
```

## Training Inpainting Model

### Preparations

Before running the training code, please make sure you have downloaded the pretrained model in "model/diffusion_and_controlnet/pretrained" from our baidupan link.

You also need to update the concept list for validation via "diffuser/inpaint/concept_list/0430/make_concept_list.ipynb".

Please also update the path in the **"diffuser/inpaint/bash/polyp/training_inpaint.sh"** file.

### Training

To train the inpainting model, you can run the following command:
```bash
cd diffuser/inpaint
nohup bash bash/script/polyp/training_inpaint.sh  > "your training log path" &
```

## Training Controlnet Model

### Preparations

Before running the training code, please make sure you have downloaded the pretrained model in "model/diffusion_and_controlnet/pretrained" from our baidupan link.

You also need to update the concept list for validation via "diffuser/controlnet/concept_list/0430/make_concept_list.ipynb".

Please also update the path in the **"diffuser/controlnet/bash/train/polyp/train_multicontrolnet.sh"** file.


### Training
To train the controlnet model, you can run the following command:
```bash
cd diffuser/controlnet
nohup bash bash/train/polyp/train_multicontrolnet.sh  > "your training log path" &
```

## Visualization Inpainting Images

If you want to visualize the inpainting images, you can open the ipynb in "diffuser/controlnet/visualization/polyp/infer_multicontrolnet_inpaint.ipynb" and run the code.

It will generating inpainting images for visualization.


## Generating Inpainting Images

To generate inpainting images, you need to modified the path in the file "diffuser/controlnet/bash/infer/polyp/infer_multiplecontrolnet.sh".

Then you can run the command below to generate inpainting images:
```bash
cd diffuser/controlnet
nohup bash bash/infer/polyp/infer_multiplecontrolnet.sh  > "your generating log path" &
```

## Training Pseudo-mask Refinement network

### Preparations

Before running the training code, you need to download the pretrained model in "model/segmentation/pretrained_pth" from our baidupan link, to the path of "Polyp-PVT/pretrained_pth".

### Training

To train the pseudo-mask refinement network, you can run the following command, after changing the log path in the **"train.sh"** file:
```bash
cd Polyp-PVT_box_guide
nohup bash bash/polyp/train.sh  > "your training log path" &
```

## Pseudo-mask Refinement

To refine the pseudo-mask of the synthetic images, you need to modified the batch_infer.sh:
```bash

python -W ignore batch_infer.py \
--images_root "the path of synthetic images" \
--coarse_mask_root "the path of inpainting region (initial/coarse mask)" \
--output_path "the root of output path" \
--condition_mask_root "the path of boundary conditions (the mask of the synthetic image)" \
--resolution 512 \
--iters 5 \
--testsize 352 \
--pth_path "the path of pseudo-mask refinement network" \
--pth_path_original "the path of baseline segmentation model" 

```

Then, run the following command, after changing the log path in the **"batch_infer.sh"** file:
```bash
cd Polyp-PVT_box_guide
nohup bash bash/polyp/refine.sh  > "your refinement log path" &

```
## Suitible cases selection

After pseudo-mask refinement, please run the "build_merge_dataset.ipynb" to generate the match data for further training.

Please change the path in the "build_merge_dataset.ipynb" to the path of your device.

For case selection, you can simply change the "align_score_cutoff" in "Polyp-PVT/bash/polyp/aug/train_aug.sh"

## Training Segmentation Model

### Preparations

Before running the training code, you need to download the pretrained model in "model/segmentation/pretrained_pth" from our baidupan link, to the path of "Polyp-PVT/pretrained_pth".

### Training

To train the baseline segmentation model, you can run the following command, after changing the log path in the **"train.sh"** file:
```bash
cd Polyp-PVT
nohup bash bash/polyp/baseline/train.sh  > "your training log path" &
``` 

To train the augmentation segmentation model, you can run the following command, after changing the log path in the **"train_aug.sh"** file:
```bash
cd Polyp-PVT
nohup bash bash/polyp/aug/train_aug.sh  > "your training log path" &
``` 

## Tesing Segmentation Model

To test the segmentation model, you can run the following command, after changing the log path in the **"test.sh"** file:
```bash
cd Polyp-PVT
nohup bash bash/polyp/baseline/test.sh  > "your testing log path" &
```

```bash
cd Polyp-PVT
nohup bash bash/polyp/aug/test.sh  > "your testing log path" &
```

## Results Analysis
To summary the results of our project, you can run the **"fairness_analysis.ipynb"**, but attention to change the path of the results. It can automatically generate the results of the bias analysis and summary the performance of the segmentation model across different datasets.