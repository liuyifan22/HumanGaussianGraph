import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
import torchvision.transforms as transforms
from PIL import Image
import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.options import AllConfigs, Options
from core.models import HGG
from avatar.test_mvdiffusion_seq import load_wonder3d_pipeline

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scipy.spatial.transform import Slerp, Rotation

# from charactergen.webui import multiview_inference
weight_dtype = torch.float16


def crop(img, length=512):
    bs, views, ch, h, w = img.shape
    if h > w:
        start = (h - w) // 2
        img_cropped = img[:, :, :, start : start + w, :]
    else:
        start = (w - h) // 2
        img_cropped = img[:, :, :, :, start : start + h]

    # Reshape to [N, C, H, W] format
    img_cropped = img_cropped.view(-1, ch, img_cropped.shape[-2], img_cropped.shape[-1])

    # Resize the image using torch.nn.functional.interpolate
    img_resized = F.interpolate(
        img_cropped, size=(length, length), mode="bilinear", align_corners=False
    )

    # Reshape back to [bs, views, channel, length, length]
    img_resized = img_resized.view(bs, views, ch, length, length)

    return img_resized


from core.options import AllConfigs

opt = tyro.cli(AllConfigs)

if opt.data_mode == "s3":
    from sys_eval.provider_mvhuman_eval import MvhumanDataset as Dataset
else:
    raise NotImplementedError
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_dataset = Dataset(opt, training=True)
eval_dataloader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0 if opt.is_debug else opt.num_workers,  # debugging
    pin_memory=True,
    drop_last=True,
)
data_iterator = iter(eval_dataloader)


# os.environ["U2NET_ONNX_RUNTIME"] = "0"
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

opt = tyro.cli(AllConfigs)
opt.cam_radius = 2.0

# model
model = HGG(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith("safetensors"):
        ckpt = load_file(opt.resume, device="cpu")
    else:
        ckpt = torch.load(opt.resume, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    print(f"[INFO] Loaded checkpoint from {opt.resume}")
else:
    print(f"[WARN] model randomly initialized, are you sure?")

# device

model = model.float().to(device)
model.eval()
import lpips

loss_fn = lpips.LPIPS(net="alex").to(device)
model.opt.cam_radius = 1.96
rays_embeddings = model.prepare_default_rays(device)


class cfg:
    # pretrained_model_name_or_path = "./avatar/ckpts"
    pretrained_model_name_or_path = "./pretrained/wonder3d-1.0"


wonder3d = False
if wonder3d:
    pipe = load_wonder3d_pipeline(cfg=cfg())
    pipe = pipe.to(device)

import datetime
import imageio
import json
from skimage.metrics import (
    structural_similarity as ssim_metric,
    peak_signal_noise_ratio as psnr_metric,
)

import torch
import glob
import numpy as np

# process function
os.makedirs("output_eval", exist_ok=True)

Train_len = 8


def process(opt: Options, video_all, time_tag):
    name = video_all["0"]["human"][0]
    print(f"[INFO] Processing {name}")

    keys = list(video_all.keys())
    video = {key: video_all[key] for key in keys[:Train_len]}
    novel_pose = {key: video_all[key] for key in keys[Train_len:]}

    workspace_name = f"output_eval/{time_tag}_{opt.workspace}"
    os.makedirs(workspace_name, exist_ok=True)
    human_folder = os.path.join(workspace_name, name)
    os.makedirs(human_folder, exist_ok=True)

    frame_ids = list(video.keys())

    for index, frame_id in enumerate(frame_ids):

        data = video[frame_id]
        pose_name = data["pose"][0]
        print(f"[INFO] Processing {name}---{pose_name}")
        cur_folder = os.path.join(human_folder, pose_name)
        os.makedirs(cur_folder, exist_ok=True)
        # proj_matrix = data["proj_matrix"][0]
        imgs_in = video[frame_id]["images_output"][0][0]  # batch_0, view_front

        if wonder3d:
            imgs_in = F.interpolate(
                imgs_in.unsqueeze(0),
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )
            first_img = imgs_in.squeeze(0)
            first_img_ori = first_img.clone()
            # Convert first_img (a torch tensor of shape [C, H, W]) to a numpy array
            img_np = first_img.cpu().permute(1, 2, 0).numpy()  # now in shape [H, W, C]
            # Assuming the image pixel values are in [0, 1], create a mask for non-white pixels
            mask = (img_np < 0.98).any(
                axis=-1
            )  # True for pixels that are not pure white
            # Find the coordinates of non-white pixels
            coords = np.argwhere(mask)
            if coords.size:
                top, left = coords.min(axis=0)
                bottom, right = coords.max(axis=0) + 1  # add 1 for an exclusive bound
            else:
                # Fallback if no man is detected; use full image dimensions
                top, left, bottom, right = 0, 0, img_np.shape[0], img_np.shape[1]

            # print(f"Detected bounding box: top-left ({left}, {top}), bottom-right ({right}, {bottom})")
            # Optionally, crop first_img to the bounding box:
            cropped = first_img[:, top:bottom, left:right]
            C, h_crop, w_crop = cropped.shape

            # Calculate center of mass for non-white pixels in the cropped area
            cropped_np = cropped.cpu().permute(1, 2, 0).numpy()  # Convert to [H, W, C]
            mask = (cropped_np < 0.98).any(axis=-1)  # Non-white pixels
            coords = np.argwhere(mask)

            if coords.size:
                # Get center of mass
                center_of_mass_y = np.mean(coords[:, 0])
                center_of_mass_x = np.mean(coords[:, 1])

                # Calculate where to place the crop so center of mass is in the middle
                paste_top = int(128 - center_of_mass_y)
                paste_left = int(128 - center_of_mass_x)

                # Ensure it stays within bounds
                paste_top = max(0, min(paste_top, 256 - h_crop))
                paste_left = max(0, min(paste_left, 256 - w_crop))
            else:
                # Fallback to center if no valid pixels found
                paste_top = (256 - h_crop) // 2
                paste_left = (256 - w_crop) // 2

            # Create background and place cropped image according to center of mass
            background = torch.ones(
                (C, 256, 256), dtype=first_img.dtype, device=first_img.device
            )
            background[
                :, paste_top : paste_top + h_crop, paste_left : paste_left + w_crop
            ] = cropped
            first_img = background

            imgs_in = (
                torch.tensor(first_img).unsqueeze(0).repeat(12, 1, 1, 1).to(device)
            )

            # NOTE: camera embeddings for wonder3d, specific to MvHumanNet dataset
            camera_embeddings = torch.tensor(
                [
                    [0.6610, 0.6610, 0.0000, 1.0000, 0.0000],
                    [0.6610, 0.6477, 0.7747, 1.0000, 0.0000],
                    [0.6610, 0.6323, 1.5441, 1.0000, 0.0000],
                    [0.6610, 0.6397, 3.0841, 1.0000, 0.0000],
                    [0.6610, 0.6627, 4.6647, 1.0000, 0.0000],
                    [0.6610, 0.6666, 5.4768, 1.0000, 0.0000],
                    [0.6610, 0.6610, 0.0000, 0.0000, 1.0000],
                    [0.6610, 0.6477, 0.7747, 0.0000, 1.0000],
                    [0.6610, 0.6323, 1.5441, 0.0000, 1.0000],
                    [0.6610, 0.6397, 3.0841, 0.0000, 1.0000],
                    [0.6610, 0.6627, 4.6647, 0.0000, 1.0000],
                    [0.6610, 0.6666, 5.4768, 0.0000, 1.0000],
                ]
            ).to(device)

            # smpl_embeddings = torch.zeros(12,10).to(device)
            # camera_embeddings = torch.cat([camera_embeddings, smpl_embeddings], dim=-1)
            out = pipe(
                imgs_in,
                camera_embeddings,
                generator=None,
                guidance_scale=3.0,
                output_type="pt",
                num_images_per_prompt=1,
            ).images

            bsz = out.shape[0] // 2

            mv_image = out[bsz:].permute(0, 2, 3, 1).cpu().numpy()

            play_path = os.path.join(cur_folder, "playground")
            os.makedirs(play_path, exist_ok=True)
            for i in range(bsz):
                imageio.imwrite(
                    f"{play_path}/{name}_{i}.png", (mv_image[i] * 255).astype(np.uint8)
                )

            mv_image = np.stack(
                [first_img_ori.permute(1, 2, 0), mv_image[4], mv_image[3], mv_image[2]],
                axis=0,
            )  # [4, 256, 256, 3], float32

            input_image = (
                torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device)
            )  # [4, 3, 256, 256]
            input_image = F.interpolate(
                input_image,
                size=(opt.input_size, opt.input_size),
                mode="bilinear",
                align_corners=False,
            )
            input_image = TF.normalize(
                input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
            )
            rays_embeddings = data["input"][0][:, 3:, ...].to(device)
            input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(
                0
            )  # [1, 4, 9, H, W]

            # change the input image to the generated images
            video[frame_id]["input"] = input_image
        else:
            print("using ground truth input")
            break

    torch.cuda.empty_cache()

    for index, frame_id in enumerate(frame_ids):
        data = video[frame_id]
        pose_name = data["pose"][0]
        pose_name_ori = pose_name
        print(f"[INFO] Processing {name}---{pose_name}")
        cur_folder = os.path.join(human_folder, pose_name)
        os.makedirs(cur_folder, exist_ok=True)
        # # proj_matrix = data["proj_matrix"][0]
        # imgs_in = video[frame_id]["images_output"][0][0] # batch_0, view_front

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float32):

                """index == Train_len-1"""
                novel_input = novel_pose
                output = model(video, extra_poses=novel_input)
                gaussians = output["gaussians"][0]
                gaussians = gaussians[index].unsqueeze(0)

            torch.cuda.empty_cache()
            # save gaussians
            model.gs.save_ply(gaussians, os.path.join(cur_folder, name + ".ply"))
            images = []

            opacity = gaussians[0, :, 3:4].contiguous().float()
            mask = opacity.squeeze(-1) >= 0.001
            gaussians = gaussians[0, mask].unsqueeze(0)

            for i in range(data["cam_view"].shape[1]):

                cam_view = data["cam_view"][:, i].to(device)
                cam_view_proj = data["cam_view_proj"][:, i].to(device)
                cam_pos = data["cam_pos"][:, i].to(device)
                # Set minimal Gaussian attributes:
                # Convention per gaussian: [xyz, opacity, scales, quaternion, color]
                # Here we create 20 gaussians with constant values.
                ori_gaussian = gaussians[0].half()
                ori_gaussian = ori_gaussian.float()
                gaussians = ori_gaussian
                gaussians = gaussians.unsqueeze(0)  # add batch dimension: [1, N, 14]

                # NOTE: This is used for MvHumanNet! Adjust parameters if needed.
                raster_settings = GaussianRasterizationSettings(
                    image_height=1024,
                    image_width=1224,
                    tanfovx=0.45678058031918056,  # adjust if needed
                    tanfovy=0.45678058031918056,  # 0.45678058031918056,
                    bg=torch.tensor(
                        [1.0, 1.0, 1.0], dtype=torch.float32, device=gaussians.device
                    ),
                    scale_modifier=1,
                    viewmatrix=cam_view[0].clone(),  # [4,4]
                    projmatrix=cam_view_proj[0].clone(),  # [4,4]
                    sh_degree=0,
                    campos=cam_pos[0].clone(),  # [3]
                    prefiltered=False,
                    debug=True,
                )
                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                # Prepare Gaussian components (from first batch sample).
                image, _, _, _ = rasterizer(
                    means3D=gaussians[0, :, 0:3].float(),
                    means2D=torch.zeros_like(
                        gaussians[0, :, 0:3],
                        dtype=torch.float32,
                        device=gaussians.device,
                    ),
                    shs=None,
                    colors_precomp=gaussians[0, :, 11:].float(),
                    opacities=gaussians[0, :, 3:4].float(),
                    scales=gaussians[0, :, 4:7].float(),
                    rotations=gaussians[0, :, 7:11].float(),
                    cov3D_precomp=None,
                )

                image = crop(image.unsqueeze(0).unsqueeze(0), 512).squeeze(0)

                # Save result as PNG.
                img_np = (
                    (image.squeeze(0).clamp(0, 1).permute(1, 2, 0).detach() * 255)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )  # 512, 512, 3
                img_pred = (
                    image.squeeze(0).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
                )

                img_gt = data["images_output"][0][i].permute(1, 2, 0).cpu().numpy()
                imageio.imwrite(
                    f"{cur_folder}/{name}_{i}_gt.png", (img_gt * 255).astype(np.uint8)
                )

                imageio.imwrite(f"{cur_folder}/{name}_{i}.png", img_np)
                images.append(np.expand_dims(img_np, axis=0))

            images = np.concatenate(images, axis=0)
            imageio.mimwrite(os.path.join(cur_folder, name + ".mp4"), images, fps=3)


time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
total_videos = len(eval_dataset)
processed_count = 0

# Process one video at a time
while processed_count < total_videos:
    # Explicitly fetch one video
    video = next(data_iterator)
    process(opt, video, time_tag)
    processed_count += 1

    # Optional: Clear CUDA cache after each video
    torch.cuda.empty_cache()

    print(f"Processed {processed_count}/{total_videos} videos")
