import enum
from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from typing import Literal, Tuple, Optional, Any
import cv2
import random
import yaml
import pickle
import json
import os, sys
import math

import PIL.Image
# from .normal_utils import trans_normal, normal2img, img2normal
import pdb

class MvhumannetDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        num_views: int,
        bg_color: Any,
        img_wh: Tuple[int, int],
        human_list: Optional[str] = None,
    ):
        self.root_dir = Path(root_dir)
        self.num_views = num_views
        self.bg_color = bg_color
        self.img_wh = img_wh
        
        # load human id, view id, pose id
        human_info = json.load(os.path.join(self.root_dir, "human.json"))
        self.human_id_list = human_info["human_id"] # e.g. 100001
        self.view_id_list = human_info["view_id_list"] # e.g. CC32871A004
        self.pose_id_list = human_info["pose_id_list"] # e.g. 0005
        self.camera_list = []
        for view_id in self.view_id_list:
            self.camera_list.appemd(self.load_camera(self.human_id_list[0], view_id))
            
        
        
    def load_image(self, human_id, view_id, pose_id):
        img_path = os.path.join(self.root_dir, human_id, view_id, "images_lr", pose_id+"_img.jpg")
        img = np.array(Image.open(img_path))
        img = img.astype(np.float32) / 255. # [0, 1]
        return img
    
    def load_mask(self, human_id, view_id, pose_id):
        mask_path = os.path.join(self.root_dir, human_id, view_id, "fmask_lr", pose_id+"_img_fmask.png")
        mask = np.array(Image.open(mask_path))
        mask = mask.astype(np.float32) / 255 # {0, 1}
        return mask
    
    def load_camera(self, human_id, view_id, pose_id=None):
        camera_scale_path = os.path.join(self.root_dir, human_id, "camera_scale.pkl")
        camera_scale = pickle.load(open(camera_scale_path, "rb"))
        
        camera_extrinsics = json.load(os.path.join(self.root_dir, human_id, "camera_extrinsics.json"))
        cam_name = "1_" + view_id
        camera_extrinsic = camera_extrinsics[cam_name]
        R = camera_extrinsic['rotation']
        T = np.array(camera_extrinsic['translation'])[:, None] / 1000 * camera_scale
        RT = np.hstack((R, T)) # (3, 4)
        return RT
    
    def cartesian_to_spherical(self, xyz):
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T # change to cam2world

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
      
        return d_theta, d_azimuth
        
    def resize(self, img, mask):
        resize_img = np.zeros((256, 256), dtype=img.dtype)
        resize_mask = np.zeros((256, 256), dtype=mask.dtype)
        hx, wy = np.where(mask > 0)
        x1, x2, x3, x4 = np.max(hx), np.min(hx), np.max(wy), np.min(wy)
        img = Image.fromarray(img[x2:x1, x4:x3])
        mask = Image.fromarray(mask[x2:x1, x4:x3])
        
        if x1 - x2 >= x3 - x4:
            resize_h = 224
            resize_w = np.around(224 * (x3 - x4) / (x1 - x2))
            new_img = np.array(img.resize([resize_h, resize_w])) # (resize_h, resize_w)
            new_mask = np.array(mask.resize([resize_h, resize_w]))
            h1, h2 = 16, 240
            w1 = (256 - resize_w) // 2
            w2 = w1 + resize_w
            resize_img[h1:h2, w1:w2] = new_img
            resize_mask[h1:h2, w1:w2] = new_mask
        else:
            resize_w = 224
            resize_h = np.around(224 * (x1 - x2) / (x3 - x4))
            new_img = np.array(img.resize([resize_h, resize_w])) # (resize_h, resize_w)
            new_mask = np.array(mask.resize([resize_h, resize_w]))
            w1, w2 = 16, 240
            h1 = (256 - resize_h) // 2
            h2 = h1 + resize_h
            resize_img[h1:h2, w1:w2] = new_img
            resize_mask[h1:h2, w1:w2] = new_mask

        return resize_img, resize_mask
    
    def __len__(self):
        return len(self.human_id_list) * len(self.pose_id_list)

    def __getitem__(self, index) -> Any:
        human_id = self.human_id_list[index // len(self.pose_id)]
        pose_id = self.pose_id_list[index % len(self.pose_id_list)]
        
        cond_view_id = self.view_id_list[0] # front view
        tgt_view_ids = self.view_id_list # all 6 views
        
        cond_w2c = self.camera_list[0]
        
        cond_img = self.load_image(human_id, cond_view_id, pose_id)
        cond_mask = self.load_mask(human_id, cond_view_id, pose_id)
        cond_img, cond_mask = self.resize(cond_img, cond_mask) # (256, 256)
        img_tensors_in = [torch.tensor(cond_img)] * self.num_views
        
        img_tensors_out = []
        elevations = []
        azimuths = []
        
        for i, tgt_view_id in enumerate(tgt_view_ids):
            tgt_img = self.load_image(human_id, tgt_view_id, pose_id)
            tgt_mask = self.load_mask(human_id, tgt_view_id, pose_id)
            tgt_img, tgt_mask = self.resize(tgt_img, tgt_mask)
            img_tensors_out.append(tgt_img)
            
            tgt_w2c = self.camera_list[i]
            elevation, azimuth = self.get_T(tgt_w2c, cond_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)
            
        img_tensors_in = torch.stack(img_tensors_in, dim=0).float() # (Nv, 3, H, W)
        img_tensors_out = torch.stack(img_tensors_out, dim=0).float() # (Nv, 3, H, W)
        
        elevations = torch.as_tensor(elevations).float().squeeze(1)
        azimuths = torch.as_tensor(azimuths).float().squeeze(1)
        elevations_cond = torch.as_tensor([0] * self.num_views).float()  # fixed only use 4 views to train    
        camera_embeddings = torch.stack([elevations_cond, elevations, azimuths], dim=-1) # (Nv, 3)    
            
            
            
        

        return super().__getitem__(index)
        
        
if __name__ == "__main__":
    # import yaml
    # with open('/dataxl/mvhuman/mvhuman_data/100001/smplx/exp.yml', 'r') as f:
    #     file_content = f.read()
    #     content = yaml.load(file_content, yaml.FullLoader)
    #     print(type(content['sub']))
    # img_path = '/dataxl/mvhuman/mvhuman_data/100001/images_lr/CC32871A004/0005_img.jpg'
    # img = np.array(Image.open(img_path))
    # img = np.float32(img > 0)
    # mask_path = '/dataxl/mvhuman/mvhuman_data/100001/fmask_lr/CC32871A004/0005_img_fmask.png'
    # mask = np.array(Image.open(mask_path))
    # hx, wy = np.where(mask > 0)
    # print(hx)
    # print(np.argmax(hx))
    # print(np.argmin(hx))
    # print(np.argmax(wy))
    # print(np.argmin(wy))
    import pickle
    # camera_scale_fn = r'/dataxl/mvhuman/mvhuman_data/100001/camera_scale.pkl'
    camera_scale_fn = r'/dataxl/mvhuman/mvhuman_24/200001/camera_scale.pkl'
    camera_scale = pickle.load(open(camera_scale_fn, "rb"))
    print(camera_scale==120/65) # true
    