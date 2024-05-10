import torch
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle, random
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler,
)
import argparse


from PIL import Image
import cv2
import argparse
import os
import numpy as np
import pandas as pd
import tqdm
import albumentations as A
import ast
from PIL.Image import Resampling


def random_size(control_image, source_images, mask, ratio = 0.5):
    # reduce the H,W up to 0.5
    ratio = random.uniform(ratio, 1.0)
    print(ratio)
    control_image = control_image.resize((int(512*ratio), int(512*ratio)), Resampling.BILINEAR)
    source_images = source_images.resize((int(512*ratio), int(512*ratio)), Resampling.BILINEAR)
    mask = mask.resize((int(512*ratio), int(512*ratio)), Resampling.BILINEAR)
    
    # padding in right and bottom
    padding = 512 - control_image.size[0]
    control_image = cv2.copyMakeBorder(np.array(control_image), 0, padding, 0, padding, cv2.BORDER_CONSTANT, value = 0)
    source_images = cv2.copyMakeBorder(np.array(source_images), 0, padding, 0, padding, cv2.BORDER_CONSTANT, value = 0)
    mask = cv2.copyMakeBorder(np.array(mask), 0, padding, 0, padding, cv2.BORDER_CONSTANT, value = 0)

    control_image = Image.fromarray(control_image)
    source_images = Image.fromarray(source_images)
    mask = Image.fromarray(mask)
    return control_image, source_images, mask




def switch_color(img1,img2):
    image = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
    image2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)

    mean , std  = image.mean(axis=(0,1), keepdims=True), image.std(axis=(0,1), keepdims=True)
    mean2, std2 = image2.mean(axis=(0,1), keepdims=True), image2.std(axis=(0,1), keepdims=True)
    image = np.uint8(np.clip((image-mean)/std*std2+mean2, 0, 255))
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    return image

def update_bbox(bbox, x_ratio, y_ratio):
    bbox = np.array(bbox)
    bbox[0] = int(bbox[0] * x_ratio)
    bbox[1] = int(bbox[1] * y_ratio)
    bbox[2] = int(bbox[2] * x_ratio)
    bbox[3] = int(bbox[3] * y_ratio)
    return bbox

def calculate_masked_rgb_mean(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    mean_color = cv2.mean(img, mask=mask)[:3]
    return mean_color

def get_patch_mean(img, i, j, patch_size):
    patch = img[i:i+patch_size, j:j+patch_size]
    return np.mean(patch, axis=(0, 1))

def split_image_and_calculate_means(img, patch_size=32):
    h, w, _ = img.shape
    patch_means = []
    patch_centers = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    for i in range(patch_size, h - patch_size, patch_size):
        for j in range(patch_size, w - patch_size, patch_size):
            center_patch_mean = get_patch_mean(img, i, j, patch_size)
            top_patch_mean = get_patch_mean(img, i - patch_size, j, patch_size) if i - patch_size >= 0 else 0
            bottom_patch_mean = get_patch_mean(img, i + patch_size, j, patch_size) if i + patch_size < h else 0
            left_patch_mean = get_patch_mean(img, i, j - patch_size, patch_size) if j - patch_size >= 0 else 0
            right_patch_mean = get_patch_mean(img, i, j + patch_size, patch_size) if j + patch_size < w else 0
            
            # 计算五个区域的平均值
            total_mean = (center_patch_mean + top_patch_mean + bottom_patch_mean + left_patch_mean + right_patch_mean) / 5
            patch_centers.append((i + patch_size // 2, j + patch_size // 2))
            patch_means.append(total_mean)
    
    return patch_means, patch_centers


def find_nearest_patch_center(target_rgb, patch_rgbs, patch_centers):
    target_rgb = np.array(target_rgb)
    patch_rgbs = np.array(patch_rgbs)
    distances = np.linalg.norm(patch_rgbs-target_rgb, axis = 1)
    nearest_idx = np.argmin(distances)
    return patch_centers[nearest_idx]

def recenter_image(img, original_center, new_center):
    # print(original_center, new_center)
    dy, dx = np.subtract(new_center, original_center)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    recentered_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return recentered_img

def get_moments(img):
    img = img.convert('L')
    img = np.array(img)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 使用 max 函数和一个简单的 lambda 函数，找到面积最大的连通区域
    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cY, cX)

def get_mask(mask, bbox):
    H, W = mask.shape
    extended_bbox=np.array(bbox)
    
    left_freespace=min(bbox[0]-0, 10)
    right_freespace=min(W-bbox[2],10)
    up_freespace=min(bbox[1]-0,10)
    down_freespace=min(H-bbox[3],10)
    
    extended_bbox[0]=bbox[0]-int(left_freespace)
    extended_bbox[1]=bbox[1]-int(up_freespace)
    extended_bbox[2]=bbox[2]+int(right_freespace)
    extended_bbox[3]=bbox[3]+int(down_freespace)
    
    bbox_mask = np.zeros_like(mask)
    bbox_mask[extended_bbox[1]:extended_bbox[3],extended_bbox[0]:extended_bbox[2]] = 1
    mask_img= mask*bbox_mask
    return mask_img

def get_largest_bbox(bboxs):
    bboxs.sort(key = lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
    return bboxs[0]

def copy_paste(original_image, source_images, mask):
    mask = np.where(np.array(mask)> 127,1 ,0).astype(np.uint8)
    mask = mask[:,:,np.newaxis]
    output = original_image.copy()
    output = output*(1 - mask) + source_images*mask
    return Image.fromarray(output)

def get_size(mask):
    mask = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY)
    mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    mask_size = np.sum(mask)
    return mask_size/(mask.shape[0]*mask.shape[1])

class PolypDataset(dataset.Dataset):
    def __init__(self, original_img_dir, bbox_dirs, mask_dirs, img_dirs, resolution, iter):
        candidates = pd.read_csv(original_img_dir)
        candidates_map = {}
        candidates['background_image_path'] = candidates['background_image_path'].apply(ast.literal_eval)
        
        for i in range(candidates.shape[0]):
            if candidates.iloc[i, 0] not in candidates_map:
                candidates_map[candidates.iloc[i, 0]] = candidates.iloc[i, 1]
                
        # background_path = original_img_dir
        bbox_dirs = bbox_dirs.split("+")
        mask_dirs = mask_dirs.split("+")
        img_dirs = img_dirs.split("+")
        
        self.bboxs = []
        self.masks = []
        self.source_images = []
        
        for bbox_dir, mask_dir, img_dir in zip(bbox_dirs, mask_dirs, img_dirs):
            bboxs_path = bbox_dir
            masks_path = mask_dir
            source_image_path = img_dir
            
            bboxs = os.listdir(bboxs_path)
            masks = os.listdir(masks_path)
            source_images = os.listdir(source_image_path)
            # backgrounds = os.listdir(background_path)
            
            bboxs.sort()
            masks.sort()
            source_images.sort()
            # backgrounds.sort()
            
            self.bboxs += [os.path.join(bboxs_path, bbox) for bbox in bboxs]
            self.masks += [os.path.join(masks_path, mask) for mask in masks]
            self.source_images += [os.path.join(source_image_path, source_image) for source_image in source_images]
        
        self.candidate_background = candidates_map
        # self.candidate_background = [os.path.join(background_path, background) for background in backgrounds]
        
        self.iter = iter
        self.resolution = resolution
        self.length = len(self.bboxs)
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5], [0.5])
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        
        img_path = self.source_images[index]
        mask_path = self.masks[index]
        bbox_path = self.bboxs[index]
        name = img_path.split('/')[-1].split('.')[0]
        
        background_path = self.candidate_background[img_path][self.iter+10]
        
        # the current bboxs is a list of bbox, we need to pick the lagrest bbox
        bbox = random.choice(pickle.load(open(bbox_path, 'rb')))

        mask = np.array(load_image(mask_path).convert('L'))
        mask = get_mask(mask, bbox)
        control_image = Image.fromarray(mask.copy()).convert('RGB')
        
        mask = cv2.dilate(mask, np.ones((30,30), np.uint8), iterations=1)
        mask = Image.fromarray(mask)

        
        original_image = load_image(background_path).convert('RGB')
        original_image = original_image.crop((0.05*original_image.size[0], 0.05*original_image.size[1], 0.95*original_image.size[0], 0.95*original_image.size[1]))

        
        source_images = load_image(img_path).convert('RGB')
        
        mask = mask.resize((512, 512), Image.BILINEAR)
        control_image = control_image.resize((512, 512), Image.BILINEAR)
        original_image = original_image.resize((512, 512), Image.BILINEAR)
        source_images = source_images.resize((512, 512), Image.BILINEAR)
        source_images_copy = source_images.copy()
        
        # initial_mask to remove the total black area
        initial_mask = (cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY) > 1).astype(np.uint8)
        
        # random illumination change
        random_brightness = A.RandomBrightness(limit=(-0.1, 0.1), p=0.5)
        original_image = random_brightness(image=np.array(original_image))['image']
        original_image = Image.fromarray(original_image)
        
        temp_size = np.sum(np.array(control_image.convert("L")) > 0)/ 512 / 512
    
        if temp_size > 0.2:
            control_image, source_images, mask = random_size(control_image, source_images, mask, ratio = 0.5)
        elif temp_size > 0.1:
            control_image, source_images, mask = random_size(control_image, source_images, mask, ratio = 0.7)
        elif temp_size > 0.05:
            control_image, source_images, mask = random_size(control_image, source_images, mask, ratio = 0.9)

        old_center = get_moments(control_image)

        mean_of_source_images = calculate_masked_rgb_mean(np.array(source_images), np.where(np.array(mask)> 127,1 ,0).astype(np.uint8))
        original_mean, original_center = split_image_and_calculate_means(np.array(original_image))
        target_center = find_nearest_patch_center(mean_of_source_images, original_mean, original_center)

        source_images = recenter_image(np.array(source_images),old_center, target_center)
        mask = recenter_image(np.array(mask), old_center, target_center)
        control_image = recenter_image(np.array(control_image), old_center, target_center)
        control_image = control_image * initial_mask[:,:,np.newaxis]
        
        if temp_size > 0.05:
            control_image = cv2.erode(control_image, np.ones((5,5), np.uint8), iterations=1)
        
        polyp_size = get_size(control_image)
        
        if polyp_size < 0.03:
            ref_image = copy_paste(np.array(original_image), source_images, mask)
        else:
            ref_image = original_image
            
        source_images = Image.fromarray(source_images)
        mask = Image.fromarray(mask)
        control_image = Image.fromarray(control_image)
        
        source_images_tensor = self.normalize(self.to_tensor(source_images))
        mask_tensor = torch.where(self.to_tensor(control_image.convert('L')) > 0.5, 1, 0)
        crop_images_tensor = source_images_tensor * mask_tensor
        
        control_image_tensor = self.to_tensor(control_image)
        
        control_tensor = torch.cat([control_image_tensor, crop_images_tensor], dim=0).unsqueeze(0)
        
        prompt = "An endoscopic image of a polyp"
        
        example = {
            "normal_image": original_image,
            "control_tensor": control_tensor,
            "control_image": control_image,
            "mask": mask,
            "source_images": source_images_copy,
            "new_center": target_center,
            "prompt": prompt,
            "name": name,
            "ref_image": ref_image,
        }
        return example
               
            
        
def collate_fn(examples):
    normal_img = [example["normal_image"] for example in examples]
    
    mask = [example["mask"] for example in examples]
    
    control_tensor = [example["control_tensor"] for example in examples]
    
    control_image = [example["control_image"] for example in examples]

    source_image = [example["source_images"] for example in examples]

    prompt = [example["prompt"] for example in examples]
    
    name = [example["name"] for example in examples]
    
    new_center = [example["new_center"] for example in examples]
    
    ref_image = [example["ref_image"] for example in examples]
    
    return {
        "normal_image": normal_img,
        "control_tensor": control_tensor,
        "control_image": control_image,
        "source_image": source_image,
        "mask": mask,
        "prompt": prompt,
        "name": name,
        "new_center": new_center,
        "ref_image": ref_image,
    }
    

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--normal_image_dir",
        type=str,
        default="/data1/mingxiaoli/datasets/Kvasir-SEG_train/hard/top_10_closest_background_images.csv",
        help="The path of a csv file for the path of candidate normal images.",
    )
    parser.add_argument(
        "--polyp_image_dir",
        type=str,
        default="/data1/mingxiaoli/datasets/sun_kvasir_infer/train/images",
        help="The directory where the polyp images are stored.",
    )
    parser.add_argument(
        "--control_image_dir",
        type=str,
        default="/data1/mingxiaoli/datasets/sun_kvasir_infer/train/masks",
        help="The directory where the control images are stored.",
    )
    parser.add_argument(
        "--bbox_dir",
        type=str,
        default="/data1/mingxiaoli/datasets/sun_kvasir_infer/train/bboxs",
        help="The directory where the bboxs are stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data1/mingxiaoli/codes/diffusers/examples/controlnet/images",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size to use for training/validation.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="number of times for data augmentation",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="dreambooth_inpaint_controlnet_mask",
        help="The name of the model to train/evaluate.",
    )
    
    return parser.parse_args(input_args)



def main(args):

    # Set seed for reproducible training
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # create folder for output images
    os.makedirs(os.path.join(args.output_dir, args.model_name), exist_ok=True)
    
    prediction_dir = os.path.join(args.output_dir, args.model_name, "images")
    masks_dir = os.path.join(args.output_dir, args.model_name, "initial_masks")
    conditions_dir = os.path.join(args.output_dir, args.model_name, "conditions")
    grids_dir = os.path.join(args.output_dir, args.model_name, "grids")
    # copy_paste_dir = os.path.join(args.output_dir, args.model_name, "copy_paste")
    # poisson_dir = os.path.join(args.output_dir, args.model_name, "poisson")
    
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)
    os.makedirs(conditions_dir, exist_ok=True)
    # os.makedirs(copy_paste_dir, exist_ok=True)
    # os.makedirs(poisson_dir, exist_ok=True)
    
    iterations = args.k

    # load the model
    controlnet = args.controlnet_model_name_or_path
    stable_diffusion_inpaint = args.pretrained_model_name_or_path
    
    controlnet = ControlNetModel.from_pretrained(controlnet, torch_dtype=torch.float16)
    
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        stable_diffusion_inpaint, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    
    generator = torch.manual_seed(args.seed)
    
    for iter in range(iterations):
        # create dataset
        dataset = PolypDataset(args.normal_image_dir, args.bbox_dir, args.control_image_dir, args.polyp_image_dir, args.resolution, iter)
        
        # create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )
        
        num_inference_steps = 50
        strength = 0.8
        
        for j, batch in enumerate(dataloader):
            ref_image = batch["ref_image"]
            original_image = batch["normal_image"]
            control_tensor = batch["control_tensor"]
            control_image = batch["control_image"]
            source_images = batch["source_image"]
            name = batch["name"]
            mask = batch["mask"]
            prompt = batch["prompt"]
            # center = batch["new_center"]
            print("prompt: ", prompt)
            
            # sampling from the model
            image = pipe(prompt, num_inference_steps=num_inference_steps, generator=generator, image=ref_image, 
                        mask_image = mask, control_image = control_tensor, strength = strength)[0]
    
    
            for i in range(len(image)):
                
                mask_temp = mask[i].convert('RGB')
                all_imgs = [original_image[i], ref_image[i], control_image[i], mask_temp,  image[i], source_images[i]]
                grid_img = Image.new('RGB', (6 * args.resolution, args.resolution))
                for index, img in enumerate(all_imgs):
                    grid_img.paste(img, (index*args.resolution, 0, index*args.resolution + args.resolution, args.resolution))
                
                os.makedirs(os.path.join(grids_dir, str(iter)), exist_ok=True)
                grid_img.save(os.path.join(grids_dir, str(iter), name[i] +  '.png'))
                
                mask_img =  mask[i]
                os.makedirs(os.path.join(masks_dir, str(iter)), exist_ok=True)
                mask_img.save(os.path.join(masks_dir, str(iter), name[i] + '.png'))
                
                os.makedirs(os.path.join(prediction_dir, str(iter)), exist_ok=True)
                image[i].save(os.path.join(prediction_dir, str(iter), name[i] + '.png'))
                
                control = control_image[i].convert('L')
                os.makedirs(os.path.join(conditions_dir, str(iter)), exist_ok=True)
                control.save(os.path.join(conditions_dir, str(iter), name[i] + '.png'))
                

if __name__ == "__main__":
    args = parse_args()
    main(args)