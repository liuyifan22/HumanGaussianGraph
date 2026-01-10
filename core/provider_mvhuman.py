import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import kiui
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter
import pickle
import json
from PIL import Image

from smpl_renderer.mvhuman_tools.visual_smpl.mytools.reader import read_smpl

from cv2 import Rodrigues
import os

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class MvhumanDataset(Dataset):

    def _warn(self):
        raise NotImplementedError(
            "this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)"
        )

    def __init__(self, opt: Options, training=True):

        self.opt = opt
        self.training = training

        # NOTE: load the list of objects for training. Readers of my repo might want to change it
        if self.opt.is_debug:
            self.root_dir = "./test_dataset"
        else:
            self.root_dir = "./mvhuman_24"
        print(f"Loading dataset from {self.root_dir}")

        human_json_dir = "../human_debug.json" if self.opt.is_debug else "../human.json"
        with open(os.path.join(self.root_dir, human_json_dir), "r") as f:  # debugging
            human_info = json.load(f)
        self.items = human_info["human_id"]  # e.g. 200001,200005,200006
        self.poses = human_info["pose_id"]  # e.g. 0005, 0050
        # naive split
        select = 1 if self.opt.is_debug else self.opt.batch_size * 4
        if self.training:
            self.items = self.items[
                :-select
            ]  # for debugging, use 1, else, self.opt.batch_size
        else:
            self.items = self.items[-select:]  # self.opt.batch_size

        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (
            self.opt.zfar - self.opt.znear
        )
        self.proj_matrix[3, 2] = -(self.opt.zfar * self.opt.znear) / (
            self.opt.zfar - self.opt.znear
        )
        self.proj_matrix[2, 3] = 1

    def __len__(self):
        ret = len(self.items) if not self.opt.is_debug else len(self.items)
        return ret

    def __getitem__(self, idx):
        """In this case, we shall load all the images from DIFFERENT FRAMES(poses) of the same human, into a single data"""
        # try:
        if self.opt.is_debug:
            human_idx = idx
            video_data = {}
            for pose_idx in range(len(self.poses)):
                frame_data = self.get_single_frame(human_idx, pose_idx)
                video_data[str(pose_idx)] = frame_data
            return video_data
            # # except:
            #     print(f'[ERROR] dataset {idx}: error loading data!')
            #     return self.__getitem__(idx + 1)
        else:
            try:
                human_idx = idx
                video_data = {}
                dvd = idx % 5 * 12 * 0
                needed_poses = list(range(4))

                needed_poses = [pose * 10 + 1 + dvd for pose in needed_poses]

                for pose_idx in needed_poses:
                    frame_data = self.get_single_frame(human_idx, pose_idx)
                    video_data[str(pose_idx)] = frame_data
                return video_data
            except:
                return self.__getitem__(random.randint(0, self.__len__() - 1))

    def get_single_frame(self, human_idx, pose_idx):
        uid = self.items[human_idx]
        pose = self.poses[pose_idx]

        results = {}
        results["human"] = uid
        results["pose"] = pose
        results["root_dir"] = self.root_dir

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        face_indices = []
        colmap_w2c = []

        vid_cnt = 0

        all_vids = [
            "22236222",
            "22236234",
            "22236236",
            "22327084",
            "22327102",
            "22327109",
            "22327113",
            "22327117",
            "22236229",
            "22236235",
            "22327073",
            "22327091",
            "22327107",
            "22327111",
            "22327116",
            "22327118",
        ]
        ortho_vids = ["22327091", "22327116", "22236236", "22327073"]

        if self.training:
            # input views are in (36, 72), other views are randomly selected
            # vids = np.random.permutation(np.arange(36, 73))[:self.opt.num_input_views].tolist() + np.random.permutation(100).tolist()
            # no it is objaverse, I use mvhumannet
            random.shuffle(all_vids)
            # assert self.opt.num_input_views-4>0, "num_input_views should be larger than 4"
            vids = ortho_vids + all_vids
        else:
            # fixed views
            vids = ortho_vids

        vid_num = 0
        for vid in vids:
            vid_num += 1
            image = self.load_image(uid, vid, pose)  # (1024, 1224, 3) in [0, 1]
            mask_ori = self.load_mask(uid, vid, pose)  # (1024, 1224) in [0,1]

            """this is adaptive resizing according to mask, but it hampers the alignment of human absolute position. just abandon it."""

            # image, mask = self.resize(image, mask_ori, 512)
            image = self.crop(image, 512)
            mask = mask_ori[:, :, np.newaxis]
            mask = self.crop(mask, 512)

            if vid_num <= self.opt.num_input_views and 0:
                index_image = self.load_index_image(uid, vid, pose)
                # index_image, _ = self.resize_nearest(index_image, mask_ori, 512)
                index_image = (index_image * 255.0).astype(np.int64)
                face_index = index_image[..., 1] * 256 + index_image[..., 2]
                face_indices.append(torch.tensor(face_index, dtype=torch.int64))
            image = torch.tensor(image, dtype=torch.float32)  # [512,512,3], in [0,1]
            mask = torch.tensor(mask, dtype=torch.float32)

            w2c = self.load_camera(uid, vid, pose)
            # TODO: you may have a different camera system
            # blender world + opencv cam --> opengl world & cam
            # c2w[1] *= -1
            # c2w[[1, 2]] = c2w[[2, 1]]
            # c2w[:3, 1:3] *= -1 # invert up and forward direction
            row_to_add = np.array([0, 0, 0, 1])
            w2c = torch.tensor(np.vstack([w2c, row_to_add]), dtype=torch.float32)
            colmap_w2c.append(w2c)
            # scale up radius to fully use the [-1, 1]^3 space!

            c2w = torch.inverse(w2c)  # this is the colmap camera
            # change the camera to opengl camera
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1  # invert up and forward direction

            # """Should not do this. but thing is already done, What can I say? Our cam_radius is 2.4 !"""
            # c2w[:3, 3] *= 1 / 1.5 # 1.5 is the default scale

            # image = image.permute(2, 0, 1) # [4, 512, 512]
            # mask = image[3:4] # [1, 512, 512]
            image = image.permute(2, 0, 1)  # [3, 512, 512]
            # mask = mask.unsqueeze(0) # [1, 512, 512]
            image = image[:3] * mask + (1 - mask)  # [3, 512, 512], to white bg

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views and self.training:
            print(
                f"[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!"
            )
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n

        images = torch.stack(images, dim=0)  # [V, C, H, W]
        masks = torch.stack(masks, dim=0)  # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0)  # [V, 4, 4]
        colmap_w2c = torch.stack(colmap_w2c, dim=0)  # [V, 4, 4]
        results["ori_colmap_w2c"] = colmap_w2c

        # cam_poses is colmap camera

        images_input = F.interpolate(
            images[: self.opt.num_input_views].clone(),
            size=(self.opt.input_size, self.opt.input_size),
            mode="bilinear",
            align_corners=False,
        )  # [V, C, H, W]
        cam_poses_input = cam_poses[: self.opt.num_input_views].clone()

        # data augmentation
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(
            images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        )

        # resize render ground-truth images, range still in [0, 1]
        results["images_output"] = F.interpolate(
            images,
            size=(self.opt.output_size, self.opt.output_size),
            mode="bilinear",
            align_corners=False,
        )  # [V, C, output_size, output_size]
        results["masks_output"] = F.interpolate(
            masks.unsqueeze(1),
            size=(self.opt.output_size, self.opt.output_size),
            mode="bilinear",
            align_corners=False,
        )  # [V, 1, output_size, output_size]

        # results['face_indices'] = torch.stack(face_indices) # a list of [512, 512] torch.tensor

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(
                cam_poses_input[i],
                self.opt.input_size,
                self.opt.input_size,
                self.opt.fovy,
            )  # [h, w, 3]
            rays_plucker = torch.cat(
                [torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1
            )  # [h, w, 6]
            rays_embeddings.append(rays_plucker)

        rays_embeddings = (
            torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous()
        )  # [V, 6, h, w]
        final_input = torch.cat(
            [images_input, rays_embeddings], dim=1
        )  # [V=4, 9, H, W]
        results["input"] = final_input

        camera_intrinsics = self.load_camera_intrincis(uid)
        camera_intrinsics = torch.tensor(
            camera_intrinsics, dtype=final_input.dtype, device=final_input.device
        )
        camera_intrinsics[0] = camera_intrinsics[0] / 2
        camera_intrinsics[1] = camera_intrinsics[1] / 2
        results["camera_intrinsics"] = camera_intrinsics

        """use mvhumannet intrinsics"""
        self.proj_matrix = self.get_projection_matrix(
            results["camera_intrinsics"], 1024, 1224, self.opt.znear, self.opt.zfar
        )

        results["proj_matrix"] = self.proj_matrix

        # cameras needed by gaussian rasterizer
        # he means changing a opengl to colmap camera
        # but we are using colmap camera right now, in data loading!
        # so we do not need to do this
        # cam_view = cam_poses

        # ori

        # [[ 6.5293e-01,  7.5741e-01,  2.0586e-03, -8.4697e-02],
        #  [-2.2090e-01,  1.8782e-01,  9.5704e-01,  8.7528e-01],
        #  [ 7.2449e-01, -6.2534e-01,  2.8995e-01,  2.4228e+00],
        #  [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1  # invert up & forward direction

        cam_view = torch.inverse(cam_poses).transpose(1, 2)  # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix  # [V, 4, 4]

        cam_pos = -cam_poses[
            :, :3, 3
        ]  # [V, 3] # it is the camera position in world space

        results["cam_view"] = cam_view
        results["cam_view_proj"] = cam_view_proj
        results["cam_pos"] = cam_pos

        results["optimized_vertices"] = self.load_optimized_smpl(uid, pose)

        optimized_params = self.load_optimized_params(uid, pose)
        results["optimized_poses"] = torch.tensor(
            optimized_params["pose"], dtype=cam_pos.dtype, device=final_input.device
        )
        results["optimized_betas"] = torch.tensor(
            optimized_params["betas"], dtype=cam_pos.dtype, device=final_input.device
        )
        results["optimized_transl"] = torch.tensor(
            optimized_params["translation"],
            dtype=cam_pos.dtype,
            device=final_input.device,
        )
        results["optimized_scale"] = torch.tensor(
            optimized_params["scale"], dtype=cam_pos.dtype, device=final_input.device
        )

        return results

    def resize(self, img, mask, length=512):
        L = length
        img_dtype = img.dtype
        mask_dtype = mask.dtype
        resize_img = np.zeros((L, L, 3), dtype=img.dtype)
        resize_mask = np.zeros((L, L), dtype=mask.dtype)
        hx, wy = np.where(mask > 0)
        img = (img * 255).astype(np.uint8)
        mask = (mask * 255).astype(np.uint8)
        x1, x2, x3, x4 = np.max(hx), np.min(hx), np.max(wy), np.min(wy)
        img = Image.fromarray(img[x2:x1, x4:x3])
        mask = Image.fromarray(mask[x2:x1, x4:x3])

        if x1 - x2 >= x3 - x4:
            resize_h = int(L * 0.875)

            resize_w = np.around(resize_h * (x3 - x4) / (x1 - x2)).astype(np.int32)
            # <PIL.Image.Image image mode=RGB size=247x779 at 0x7F4429872D00>
            # PIL images are (width, height)
            new_img = np.array(img.resize([resize_w, resize_h]))  # (resize_h, resize_w)
            new_mask = np.array(mask.resize([resize_w, resize_h]))
            h1, h2 = L // 16, L - L // 16
            w1 = (L - resize_w) // 2
            w2 = w1 + resize_w
            resize_img[h1:h2, w1:w2] = new_img
            resize_mask[h1:h2, w1:w2] = new_mask
        else:
            resize_w = int(L * 0.875)
            resize_h = np.around(resize_w * (x1 - x2) / (x3 - x4)).astype(np.int32)
            new_img = np.array(img.resize([resize_w, resize_h]))  # (resize_h, resize_w)
            new_mask = np.array(mask.resize([resize_w, resize_h]))
            w1, w2 = L // 16, L - L // 16
            h1 = (L - resize_h) // 2
            h2 = h1 + resize_h
            resize_img[h1:h2, w1:w2] = new_img
            resize_mask[h1:h2, w1:w2] = new_mask

        resize_img = resize_img.astype(img_dtype) / 255.0
        resize_mask = (resize_mask / 255.0).astype(mask_dtype)
        return resize_img, resize_mask

    def crop(self, img, length=512):
        h, w, ch = img.shape
        if h > w:
            start = (h - w) // 2
            img_cropped = img[start : start + w, :, :]

        else:
            start = (w - h) // 2
            img_cropped = img[:, start : start + h, :]
        if ch == 1:
            img_resized = cv2.resize(
                img_cropped, (length, length), interpolation=cv2.INTER_NEAREST
            )
        else:
            img_resized = cv2.resize(
                img_cropped, (length, length), interpolation=cv2.INTER_LINEAR
            )

        return img_resized

    def load_image(self, human_id, view_id, pose_id):
        img_path = os.path.join(
            self.root_dir, human_id, "images_lr", view_id, pose_id + "_img.jpg"
        )
        # print(img_path)
        img = np.array(Image.open(img_path))
        img = img.astype(np.float32) / 255.0  # [0, 1]
        return img

    def load_index_image(self, human_id, view_id, pose_id):
        img_path = os.path.join(
            self.root_dir, human_id, "render_smplx", view_id, pose_id + "_indices.png"
        )
        # print(img_path)
        img = np.array(Image.open(img_path)) / 255.0
        return img

    def load_smplx_vertices(self, human_id, pose_id, device):
        pose_int = int(pose_id) // 5 - 1
        pose_id = f"{pose_int:06d}"
        vertices_path = os.path.join(
            self.root_dir, human_id, "smplx/smplx_mesh", pose_id + ".obj"
        )

        # Initialize an empty list to store the coordinates
        vertices = []

        # Open the file and read the lines
        with open(vertices_path, "r") as file:
            for line in file:
                if line.startswith("v "):
                    # Split the line into components and convert to float
                    _, x, y, z = line.split()
                    vertices.append([float(x), float(y), float(z)])

        # Convert the list of vertices to a torch tensor
        vertices_tensor = torch.tensor(vertices, device=device)
        # Ensure the tensor has the shape [10000, 3]

        # some smplx lacks the point 8559.
        if vertices_tensor.shape == (10474, 3):
            vertices_tensor1 = vertices_tensor[:8558]
            vertices_tensor2 = vertices_tensor[8558:]
            vertices_tensor_mid = (
                (vertices_tensor1[-1] + vertices_tensor2[0]) / 2
            ).unsqueeze(0)
            vertices_tensor = torch.cat(
                [vertices_tensor1, vertices_tensor_mid, vertices_tensor2], dim=0
            )
        assert vertices_tensor.shape == (10475, 3), "The tensor shape is incorrect"
        # print(vertices_tensor)

        return vertices_tensor

    def load_smpl(self, human_id, pose_id):
        pose_int = int(pose_id) // 5 - 1
        pose_id = f"{pose_int:06d}"
        smpl_path = os.path.join(
            self.root_dir, human_id, "smpl_param", pose_id + ".pkl"
        )

        while (not os.path.exists(smpl_path)) and (pose_int > 0):
            pose_int = pose_int - 1
            pose_id = f"{pose_int:06d}"
            smpl_path = os.path.join(
                self.root_dir, human_id, "smpl_param", pose_id + ".pkl"
            )
        with open(smpl_path, "rb") as f:
            data = pickle.load(f)
        vertices = data["vertices"][0]

        assert vertices.shape == (6890, 3), "The tensor shape is incorrect"

        return vertices

    def load_optimized_smpl(self, human_id, pose_id):
        smpl_path = os.path.join(
            self.root_dir, human_id, "optimized_smpl_points", pose_id + ".ply"
        )

        def read_ply_points(ply_path):
            """
            Read a PLY file (in ascii format) and return a numpy array of vertex positions.

            Parameters:
                ply_path (str): Path to the PLY file.

            Returns:
                numpy.ndarray: Array of shape (num_vertices, 3)
            """
            with open(ply_path, "r") as f:
                lines = f.readlines()

            num_vertices = 0
            header_end = 0
            # Parse header to get vertex count and header length.
            for i, line in enumerate(lines):
                if line.startswith("element vertex"):
                    num_vertices = int(line.strip().split()[-1])
                if line.strip() == "end_header":
                    header_end = i + 1
                    break

            vertices = []
            for i in range(header_end, header_end + num_vertices):
                parts = lines[i].strip().split()
                # Parse the first 3 elements as x, y, z.
                x, y, z = map(float, parts[:3])
                vertices.append([x, y, z])
            return np.array(vertices)

        vertices = read_ply_points(smpl_path)

        assert vertices.shape == (6890, 3), "The tensor shape is incorrect"

        return vertices

    def load_optimized_params(self, human_id, pose_id):
        params_path = os.path.join(
            self.root_dir, human_id, "optimized_smpl_points", pose_id + ".json"
        )
        with open(params_path, "r") as f:
            data = json.load(f)
        return data

    def load_mask(self, human_id, view_id, pose_id):
        mask_path = os.path.join(
            self.root_dir, human_id, "fmask_lr", view_id, pose_id + "_img_fmask.png"
        )
        mask = np.array(Image.open(mask_path))
        mask = mask.astype(np.float32) / 255.0  # {0, 1}
        return mask

    def load_camera(self, human_id, view_id, pose_id=None):
        camera_scale_path = os.path.join(self.root_dir, human_id, "camera_scale.pkl")

        try:
            camera_scale = pickle.load(open(camera_scale_path, "rb"))
        except:
            camera_scale = 1.5384615384615385
        # camera_extrinsics = json.load(os.path.join(self.root_dir, human_id, "camera_extrinsics.json"))
        with open(
            os.path.join(
                os.path.join(self.root_dir, human_id, "camera_extrinsics.json")
            ),
            "r",
        ) as f:
            camera_extrinsics = json.load(f)
        cam_name = "0_0_" + view_id + ".png"
        if cam_name not in camera_extrinsics:
            cam_name = "0_1_" + view_id + ".png"
        camera_extrinsic = camera_extrinsics[cam_name]
        R = camera_extrinsic["rotation"]
        # T = np.array(camera_extrinsic['translation'])[:, None]* camera_scale
        T = np.array(camera_extrinsic["translation"])[:, None] / 1000 * camera_scale
        RT = np.hstack((R, T))  # (3, 4)
        return RT

    def load_camera_intrincis(self, human_id, view_id=None):
        camera_intrinsics_path = os.path.join(
            self.root_dir, human_id, "camera_intrinsics.json"
        )
        with open(camera_intrinsics_path, "r") as f:
            camera_intrinsics = json.load(f)
        value = camera_intrinsics["intrinsics"]
        return value

    def get_projection_matrix(
        self, camera_intrinsics, image_width, image_height, znear, zfar
    ):
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]

        # Compute the perspective projection matrix
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        proj_matrix[0, 0] = 2 * fx / image_width
        proj_matrix[1, 1] = 2 * fy / image_width
        proj_matrix[0, 2] = 1 - 2 * cx / image_width
        proj_matrix[1, 2] = 2 * cy / image_width - 1
        proj_matrix[2, 2] = -(zfar + znear) / (zfar - znear)
        proj_matrix[2, 3] = 1
        proj_matrix[3, 2] = -2 * zfar * znear / (zfar - znear)

        return proj_matrix
