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
        # human_list: Optional[str] = None,
        object_list: str,
        groups_num: int=1,
        validation: bool = False,
        data_view_num: int = 6,
        num_validation_samples: int = 1,
        num_samples: Optional[int] = None,
        invalid_list: Optional[str] = None,
        trans_norm_system: bool = True,   # if True, transform all normals map into the cam system of front view
        augment_data: bool = False,
        read_normal: bool = True,
        read_color: bool = False,
        read_depth: bool = False,
        read_mask: bool = True,
        mix_color_normal: bool = False,
        suffix: str = 'png',
        subscene_tag: int = 3,
        backup_scene: str = None
    ):
        self.mix_color_normal = mix_color_normal
        self.root_dir = Path(root_dir)
        self.num_views = num_views
        self.bg_color = bg_color
        self.img_wh = img_wh
        self.read_normal = read_normal
        self.read_color = read_color
        self.read_depth = read_depth
        self.read_mask = read_mask
        
        # load human id, view id, pose id
        with open(os.path.join(self.root_dir, "human_debug.json"), 'r') as f:
            human_info = json.load(f)
        self.human_id_list = human_info["human_id"] # e.g. 100001
        
        # if validation, only use all poses of the last human (2400/50=48 poses)
        if not validation:
            self.human_id_list = self.human_id_list[:-num_validation_samples]
        else:
            self.human_id_list = self.human_id_list[-num_validation_samples:]
        if num_samples is not None:
            self.human_id_list = self.human_id_list[:num_samples]
        
        self.view_id_list = human_info["view_id"] # e.g. CC32871A004
        self.pose_id_list = human_info["pose_id"] # e.g. 0005
        self.camera_list = []
        for view_id in self.view_id_list:
            self.camera_list.append(self.load_camera(self.human_id_list[0], view_id))
        print(f">>> Total length of dataset: {len(self.human_id_list) * len(self.pose_id_list)}")
        
        
    def load_image(self, human_id, view_id, pose_id):
        img_path = os.path.join(self.root_dir, human_id, "images_lr", view_id, pose_id+"_img.jpg")
        # print(img_path)
        img = np.array(Image.open(img_path))
        img = img.astype(np.float32) / 255. # [0, 1]
        return img
    
    def load_normal(self, human_id, view_id, pose_id):
        normal_path = os.path.join(self.root_dir, human_id, "normal_maps", view_id, pose_id+"_img.jpg")
        normal = np.array(Image.open(normal_path))
        normal = normal.astype(np.float32) / 255
        return normal
    
    def load_mask(self, human_id, view_id, pose_id):
        mask_path = os.path.join(self.root_dir, human_id,  "fmask_lr",view_id, pose_id+"_img_fmask.png")
        mask = np.array(Image.open(mask_path))
        mask = mask.astype(np.float32) / 255 # {0, 1}
        return mask
    
    def load_smpl(self, human_id, pose_id):
        pose_id_int = int(pose_id)
        pose_id_divided = pose_id_int // 5 - 1
        pose_id_six_digit = f"{pose_id_divided:06d}"
        smpl_json_path = os.path.join(self.root_dir, human_id, "smplx", "smpl", pose_id_six_digit + ".json")
        with open(smpl_json_path, 'r') as f:
            smpl_info = json.load(f)[0]
        smpl_shape = np.array(smpl_info['shapes'])
        smpl_pose = np.array(smpl_info['poses'])
        return smpl_shape, smpl_pose

    
    def load_camera(self, human_id, view_id, pose_id=None):
        camera_scale_path = os.path.join(self.root_dir, human_id, "camera_scale.pkl")
        camera_scale = pickle.load(open(camera_scale_path, "rb"))
        
        # camera_extrinsics = json.load(os.path.join(self.root_dir, human_id, "camera_extrinsics.json"))
        with open(os.path.join(os.path.join(self.root_dir, human_id, "camera_extrinsics.json")), 'r') as f:
            camera_extrinsics = json.load(f)
        cam_name = "0_0_" + view_id + ".png"
        if cam_name not in camera_extrinsics:
            cam_name = "0_1_" + view_id + ".png"
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
        img_dtype=img.dtype
        mask_dtype=mask.dtype
        resize_img = np.zeros((256, 256, 3), dtype=img.dtype)
        resize_mask = np.zeros((256, 256), dtype=mask.dtype)
        hx, wy = np.where(mask > 0)
        img = (img*255).astype(np.uint8)
        mask = (mask*255).astype(np.uint8)
        x1, x2, x3, x4 = np.max(hx), np.min(hx), np.max(wy), np.min(wy)
        img = Image.fromarray(img[x2:x1, x4:x3])
        mask = Image.fromarray(mask[x2:x1, x4:x3])
        
        if x1 - x2 >= x3 - x4:
            resize_h = 224
            resize_w = np.around(224 * (x3 - x4) / (x1 - x2)).astype(np.uint8)
            # <PIL.Image.Image image mode=RGB size=247x779 at 0x7F4429872D00>
            # PIL images are (width, height)
            new_img = np.array(img.resize([resize_w, resize_h])) # (resize_h, resize_w)
            new_mask = np.array(mask.resize([resize_w, resize_h]))
            h1, h2 = 16, 240
            w1 = (256 - resize_w) // 2
            w2 = w1 + resize_w
            resize_img[h1:h2, w1:w2] = new_img
            resize_mask[h1:h2, w1:w2] = new_mask
        else:
            resize_w = 224
            resize_h = np.around(224 * (x1 - x2) / (x3 - x4)).astype(np.uint8)
            new_img = np.array(img.resize([resize_w, resize_h])) # (resize_h, resize_w)
            new_mask = np.array(mask.resize([resize_w, resize_h]))
            w1, w2 = 16, 240
            h1 = (256 - resize_h) // 2
            h2 = h1 + resize_h
            resize_img[h1:h2, w1:w2] = new_img
            resize_mask[h1:h2, w1:w2] = new_mask

        resize_img = resize_img.astype(img_dtype) / 255.
        resize_mask = (resize_mask / 255.).astype(mask_dtype)
        return resize_img, resize_mask
    
    def __len__(self):
        return len(self.human_id_list) * len(self.pose_id_list)

    def mask_image(self, img, mask, bg_color):
        img = img * mask[:, :, None] + (1 - mask[:, :, None]) * bg_color
        return img


    def __getitem_mix__(self, index) -> Any:
        # try:
        human_id = self.human_id_list[index // len(self.pose_id_list)]
        pose_id = self.pose_id_list[index % len(self.pose_id_list)]
        
        cond_view_id = self.view_id_list[0] # front view
        tgt_view_ids = self.view_id_list # all 6 views
        
        cond_w2c = self.camera_list[0]
        
        cond_img = self.load_image(human_id, cond_view_id, pose_id)
        cond_mask = self.load_mask(human_id, cond_view_id, pose_id)
        cond_img, cond_mask = self.resize(cond_img, cond_mask) # (256, 256)
        masked_cond_img = self.mask_image(cond_img, cond_mask, self.bg_color)
        # masked_PIL = Image.fromarray((masked_cond_img * 255).astype(np.uint8))
        # masked_PIL.save("masked_PIL.jpg")
        img_tensors_in = [torch.tensor(masked_cond_img).permute(2,0,1)] * self.num_views
        
        smpl_shape, smpl_pose = self.load_smpl(human_id, pose_id)
        
        
        img_tensors_out = []
        elevations = []
        azimuths = []
        
        for i, tgt_view_id in enumerate(tgt_view_ids):
            
            tgt_mask_0 = self.load_mask(human_id, tgt_view_id, pose_id)
            if self.read_normal:
                tgt_normal = self.load_normal(human_id, tgt_view_id, pose_id)
                tgt_normal, tgt_mask = self.resize(tgt_normal, tgt_mask_0)
                # need to add mask to image
                masked_tgt_img = self.mask_image(tgt_normal, tgt_mask, self.bg_color)
            else:
                tgt_img = self.load_image(human_id, tgt_view_id, pose_id)
                tgt_img, tgt_mask = self.resize(tgt_img, tgt_mask_0)
                masked_tgt_img = self.mask_image(tgt_img, tgt_mask, self.bg_color)
            
            img_tensors_out.append(torch.tensor(masked_tgt_img).permute(2, 0, 1)) # (3, H, W)            
            
            # masked_PIL = Image.fromarray((masked_tgt_img * 255).astype(np.uint8))
            # masked_PIL.save(f"masked_tgt_{i}.jpg")
            
            
            tgt_w2c = self.camera_list[i]
            elevation, azimuth = self.get_T(tgt_w2c, cond_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)
            
        img_tensors_in = torch.stack(img_tensors_in, dim=0).float() # (Nv, 3, H, W)
        img_tensors_out = torch.stack(img_tensors_out, dim=0).float() # (Nv, 3, H, W)


        elevations = np.array(elevations)
        azimuths = np.array(azimuths)

        # Convert numpy arrays to tensors
        elevations = torch.as_tensor(elevations).float().squeeze(1)
        azimuths = torch.as_tensor(azimuths).float().squeeze(1)
        
        # azimuths should be a relative value, cond_azimuth substracted
        azimuths = (azimuths - azimuths[0]) % (2 * math.pi)
        elevations_cond = torch.as_tensor([elevations[0]] * self.num_views).float()  # fixed only use 4 views to train    
        camera_embeddings = torch.stack([elevations_cond, elevations, azimuths], dim=-1) # (Nv, 3)    

        color_class = torch.tensor([0, 1]).float()
        color_task_embeddings = torch.stack([color_class]*self.num_views, dim=0)  # (Nv, 2)
        normal_class = torch.tensor([1, 0]).float()
        normal_task_embeddings = torch.stack([normal_class]*self.num_views, dim=0)  # (Nv, 2)
        return {
            'elevations_cond': elevations_cond,
            'elevations_cond_deg': torch.rad2deg(elevations_cond),
            'elevations': elevations,
            'azimuths': azimuths,
            'elevations_deg': torch.rad2deg(elevations),
            'azimuths_deg': torch.rad2deg(azimuths),
            'imgs_in': img_tensors_in,
            'imgs_out': img_tensors_out,
            'camera_embeddings': camera_embeddings,
            'task_embeddings': normal_task_embeddings if self.read_normal else color_task_embeddings,
            'smpl_shape': torch.tensor(smpl_shape).float().to(device = img_tensors_in.device),
            'smpl_pose': torch.tensor(smpl_pose).float().to(device = img_tensors_in.device)
        }
        # except:
        #     print(f">>> Error in index {index}, human_id: {human_id}, pose_id: {pose_id}")
        #     return self.__getitem__(index+1)
        
    def __getitem_joint__(self, index) -> Any:
        human_id = self.human_id_list[index // len(self.pose_id_list)]
        pose_id = self.pose_id_list[index % len(self.pose_id_list)]
        
        cond_view_id = self.view_id_list[0] # front view
        tgt_view_ids = self.view_id_list # all 6 views
        
        cond_w2c = self.camera_list[0]
        
        cond_img = self.load_image(human_id, cond_view_id, pose_id)
        cond_mask = self.load_mask(human_id, cond_view_id, pose_id)
        cond_img, cond_mask = self.resize(cond_img, cond_mask) # (256, 256)
        masked_cond_img = self.mask_image(cond_img, cond_mask, self.bg_color)
        # masked_PIL = Image.fromarray((masked_cond_img * 255).astype(np.uint8))
        # masked_PIL.save("masked_PIL.jpg")
        img_tensors_in = [torch.tensor(masked_cond_img).permute(2,0,1)] * self.num_views
        
        smpl_shape, smpl_pose = self.load_smpl(human_id, pose_id)
        
        img_tensors_out = []
        normal_tensors_out = []
        elevations = []
        azimuths = []
        
        for i, tgt_view_id in enumerate(tgt_view_ids):
            
            tgt_mask_0 = self.load_mask(human_id, tgt_view_id, pose_id)
            
            tgt_normal = self.load_normal(human_id, tgt_view_id, pose_id)
            tgt_normal, tgt_mask = self.resize(tgt_normal, tgt_mask_0)
            # need to add mask to image
            masked_tgt_normal = self.mask_image(tgt_normal, tgt_mask, self.bg_color)
            normal_tensors_out.append(torch.tensor(masked_tgt_normal).permute(2, 0, 1)) # (3, H, W)
            
            tgt_img = self.load_image(human_id, tgt_view_id, pose_id)
            tgt_img, tgt_mask = self.resize(tgt_img, tgt_mask_0)
            masked_tgt_img = self.mask_image(tgt_img, tgt_mask, self.bg_color)
            
            img_tensors_out.append(torch.tensor(masked_tgt_img).permute(2, 0, 1)) # (3, H, W)            
            
            # masked_PIL = Image.fromarray((masked_tgt_img * 255).astype(np.uint8))
            # masked_PIL.save(f"masked_tgt_{i}.jpg")
            
            
            tgt_w2c = self.camera_list[i]
            elevation, azimuth = self.get_T(tgt_w2c, cond_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)
            
        img_tensors_in = torch.stack(img_tensors_in, dim=0).float() # (Nv, 3, H, W)
        img_tensors_out = torch.stack(img_tensors_out, dim=0).float() # (Nv, 3, H, W)
        normal_tensors_out = torch.stack(normal_tensors_out, dim=0).float() # (Nv, 3, H, W)

        elevations = np.array(elevations)
        azimuths = np.array(azimuths)

        # Convert numpy arrays to tensors
        elevations = torch.as_tensor(elevations).float().squeeze(1)
        azimuths = torch.as_tensor(azimuths).float().squeeze(1)
        
        # azimuths should be a relative value, cond_azimuth substracted
        azimuths = (azimuths - azimuths[0]) % (2 * math.pi)
        elevations_cond = torch.as_tensor([elevations[0]] * self.num_views).float()  # fixed only use 4 views to train    
        camera_embeddings = torch.stack([elevations_cond, elevations, azimuths], dim=-1) # (Nv, 3)    

        color_class = torch.tensor([0, 1]).float()
        color_task_embeddings = torch.stack([color_class]*self.num_views, dim=0)  # (Nv, 2)
        normal_class = torch.tensor([1, 0]).float()
        normal_task_embeddings = torch.stack([normal_class]*self.num_views, dim=0)  # (Nv, 2)
        return {
            'elevations_cond': elevations_cond,
            'elevations_cond_deg': torch.rad2deg(elevations_cond),
            'elevations': elevations,
            'azimuths': azimuths,
            'elevations_deg': torch.rad2deg(elevations),
            'azimuths_deg': torch.rad2deg(azimuths),
            'imgs_in': img_tensors_in,
            'imgs_out': img_tensors_out,
            'normals_out': normal_tensors_out,
            'camera_embeddings': camera_embeddings,
            'normal_task_embeddings': normal_task_embeddings,
            'color_task_embeddings': color_task_embeddings,
            'smpl_shape': torch.tensor(smpl_shape).float().to(device = img_tensors_in.device),
            'smpl_pose': torch.tensor(smpl_pose).float().to(device = img_tensors_in.device)
        }
        
    def __getitem__(self, index):
        try:
            if self.mix_color_normal:
                data = self.__getitem_mix__(index)
            else:
                data = self.__getitem_joint__(index)
            return data
        except:
            print("load error ", self.all_objects[index%len(self.all_objects)] )
            if self.mix_color_normal:
                data = self.__getitem_mix__(0)
            else:
                data = self.__getitem_joint__(0)
            return data
        
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
    
    # import pickle
    # camera_scale_fn = r'/dataxl/mvhuman/mvhuman_data/100001/camera_scale.pkl'
    # camera_scale = pickle.load(open(camera_scale_fn, "rb"))
    # print(camera_scale==120/65)
    
    dataset = MvhumannetDataset(root_dir='./mvhumannet/mvhuman_24', num_views=6, bg_color=[1, 1, 1], img_wh=(256, 256))
    print(len(dataset))
    dummy = dataset[2]