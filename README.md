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

## Dataset & Model
The dataset and model we use in this project are available at baidupan:

链接：https://pan.baidu.com/s/1ku1pImibv5yhpwzebvT1WQ?pwd=czzk 
提取码：czzk 
--来自百度网盘超级会员V6的分享

### Dataset
In this link, the "data.zip" file contains all of the dataset we used and synthetic images that we generated in this project.

"data/SUN-SEG_10/TrainDataset_10" contains the training set we used for baseline segmentation model training and inpainting model training.

"data/SUN-SEG_10/SUN-SEG_10_val" contains the validation set we used for baseline segmentation model training and inpainting model training.

"data/SUN-SEG_10_test" contains the test set we used for internal segmentation model testing set.

"data/CVC-300", "data/CVC-ClinicDB", "data/Kvasir-SEG", "data/ETIS-LaribPolypDB" are external segmentation model testing sets.

"data/negatives" contains the negative background images we used for inpainting.

"data/SUN-SEG_10/multiple_controlnet_inpaint" contains all of the synthetic images we generated in this project, although we 
generate 5 folds, only use the 1st fold for further training. 
- "data/SUN-SEG_10/multiple_controlnet_inpaint/images" contain the synthetic images
- "data/SUN-SEG_10/multiple_controlnet_inpaint/images" contain the synthetic images

### Model