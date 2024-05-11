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

The high-level idea of our work is to leverage inpainting based augmentation method as a do-operator for causal and non-causal factor, which can help to break their association and enhance the generalization and reducing bias simutaneously.
![image](https://github.com/497662892/Ai-Security-Final-Project/blob/main/imgs/casual.PNG)

The framework of our proposed inpainting method is shown below:
![image](https://github.com/497662892/Ai-Security-Final-Project/blob/main/imgs/inpaint_model.PNG)

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

"model/diffusion_and_controlnet/pretrained" contrains the pretrained inpainting model and controlnet modules, which is used in finetuning the inpainting model. (no need to download if you don't want to finetune the inpainting model)

"model/diffusion_and_controlnet/finetuned" contrains the finetuned inpainting model and controlnet modules, which is used in generating synthetic images.

- "model/diffusion_and_controlnet/finetuned/inpaint_1e5" is the finetuned stable diffusion backbone model.
- "model/diffusion_and_controlnet/finetuned/boundary_controlnet" is the finetuned boundary controlnet model.
- "model/diffusion_and_controlnet/finetuned/surface_controlnet" is the finetuned surface controlnet model.

## Environment

## Training Inpainting Model

## Visualization Inpainting Images

## Generating Inpainting Images

## Training Pseudo-mask Refinement network

## Pseudo-mask Refinement

## Suitible cases selection

## Training Segmentation Model

## Tesing Segmentation Model

## Results Analysis
To summary the results of our project, you can run the **"fairness_analysis.ipynb"**, but attention to change the path of the results. It can automatically generate the results of the bias analysis and summary the performance of the segmentation model across different datasets.