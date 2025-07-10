import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import kiui
from kiui.lpips import LPIPS

from core.unet import UNet
from core.options import Options
from core.gs import GaussianRenderer
import os
import time
from torch.nn.utils.rnn import pad_sequence 
import torch_cluster
import smplx

from smpl_renderer.mvhuman_tools.visual_smpl.mytools.smplmodel import load_model
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, quaternion_invert, quaternion_multiply
# from smpl_renderer.mvhuman_tools.visual_smpl.mytools.smplmodel import load_model
# from calculate_rotation import compute_face_rotation_matrices_parallel

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None):
        # query: (L, N, E), key/value: (S, N, E)
        attn_output, _ = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        # Residual and normalization
        out = self.norm(query + self.dropout(attn_output))
        return out

class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, dropout=0.1):
        super(FFN, self).__init__()
        hidden_dim = hidden_dim or embed_dim * 4
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()  # or nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        # Residual connection and normalization.
        x = self.norm(x + residual)
        return x
    
class IntraMeshTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(IntraMeshTransformer, self).__init__()
        self.attn = CrossAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = FFN(embed_dim, dropout=dropout)
        
    def forward(self, mesh_queries, mesh_keys, attn_mask=None):
        """
        Input:
            mesh_queries: [query_length, 6890, dim], 
            mesh_keys: [key_length, 6890, dim]
            attn_mask: [6890, key_length]
        
        Output:
            mesh_queries: [query_length, 6890, dim]
        """
        mesh_queries = self.attn(mesh_queries, mesh_keys, mesh_keys, attn_mask)
        mesh_queries = self.ffn(mesh_queries)
        return mesh_queries

class InterMeshTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, connectivity_graph, connectivity_mask, batch_idx, dropout=0.1):
        super(InterMeshTransformer, self).__init__()
        self.attn = CrossAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = FFN(embed_dim, dropout=dropout)
        self.connectivity_graph = connectivity_graph
        self.connectivity_mask = connectivity_mask
        self.batch_idx = batch_idx
        
    def forward(self, mesh_queries):
        """
        Input:
            mesh_queries: [query_length, 6890, dim], 
            cross_mesh_queries: [query_length, 6890, dim]
        
        Output:
            mesh_queries: [query_length, 6890, dim]
        """
        neighbor_features = mesh_queries[self.batch_idx, self.connectivity_graph, :]  # [5, 6890, 18, channel]
        query_length, vertices, neighborhood_trunc, dim = neighbor_features.shape
        mesh_queries = mesh_queries.reshape(-1, dim).unsqueeze(1).permute(1,0,2) # [5*6890, 1, 128]-> [1, 5*6890, 128]
        neighbor_features = neighbor_features.reshape(-1,neighborhood_trunc, dim).permute(1,0,2) # [5*6890, 18, 128] -> [18, 5*6890, 128]
        connectivity_mask = self.connectivity_mask.reshape(-1,neighborhood_trunc) # [5*6890, 18]-> [18, 5*6890]
        
        mesh_queries = self.attn(mesh_queries, neighbor_features, neighbor_features, connectivity_mask)
        mesh_queries = mesh_queries.squeeze(0).reshape(query_length, vertices, dim) # [5*6890, 128] -> [5, 6890, 128]
        mesh_queries = self.ffn(mesh_queries)
        return mesh_queries

class HGG(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        self.vertices = 6890 # the number of smplx faces
        self.gaussian_dim = 128
        self.eff_dims = self.gaussian_dim 
        self.query_length = 20
        # self.key_length = 200
        self.neighbourhood_trunc = 18
        self.max_gaussian_per_frame_per_vertex = 5 # 5*10

        # f 4 2 1
        # f 8 6 5
        # f 13 15 14
        # f 17 19 18
        # f 32 30 29
        # f 36 34 33
        # f 37 39 38
        with open("./vertex_connectivity_tensors.pt", "rb") as f:
            conn_data = torch.load(f)
        self.connectivity_graph = conn_data["neighbors"]
        self.connectivity_mask = conn_data["mask"]
        
        # unet
        self.unet = UNet(
            9, 14, 
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
        )
        # freeze unet
        for param in self.unet.parameters():
            param.requires_grad = False

        # self.post_unet = nn.Linear(self.opt.up_channels[-1], self.eff_dims)
        self.relu = nn.ReLU()
        
        """initialize connectivity graph"""
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device = torch.device(f"cuda:{current_device}")
        else:
            device = torch.device("cpu")
        connectivity_graph = self.connectivity_graph.clamp(0, 3000000).repeat(self.query_length,1,1) # [5, 6890, 30]
        connectivity_graph = connectivity_graph[...,:self.neighbourhood_trunc].to(device) # [5, 6890, 18]
        connectivity_mask = self.connectivity_mask[...,:self.neighbourhood_trunc].repeat(self.query_length,1,1) # [5, 6890, 18]
        batch_idx = torch.arange(self.query_length).view(self.query_length, 1, 1).expand(self.query_length, self.vertices, self.neighbourhood_trunc).to(device)
        connectivity_mask = connectivity_mask.reshape(-1,self.neighbourhood_trunc).to(device) # [5*6890, 18]-> [18, 5*6890]
        model_path = './smpl_renderer/mvhuman_tools/visual_smpl/smpl_'
        self.smpl_model = smplx.create(gender='neutral', model_type='smpl',model_path=model_path)
        self.smpl_model.transl.requires_grad = False

        # Freeze smpl_model.body_pose
        self.smpl_model.body_pose.requires_grad = False

        # Freeze smpl_model.global_orient
        self.smpl_model.global_orient.requires_grad = False

        # Freeze smpl_model.betas
        self.smpl_model.betas.requires_grad = False

        self.mesh_queries = nn.Parameter(torch.randn(self.vertices, self.query_length, self.gaussian_dim, dtype=torch.float32))
        # self.mesh_keys = nn.Parameter(torch.randn(self.faces, 1, self.gaussian_dim, dtype=torch.float32) * 0.1)
        # you can adjust as you like
        self.intra_mesh_transformer_1 = IntraMeshTransformer(self.gaussian_dim, num_heads=4)
        self.inter_mesh_transformer_1 = InterMeshTransformer(self.gaussian_dim, num_heads=4, connectivity_graph=connectivity_graph, connectivity_mask=connectivity_mask, batch_idx=batch_idx)
        self.intra_mesh_transformer_2 = IntraMeshTransformer(self.gaussian_dim, num_heads=4)
        self.inter_mesh_transformer_2 = InterMeshTransformer(self.gaussian_dim, num_heads=4, connectivity_graph=connectivity_graph, connectivity_mask=connectivity_mask, batch_idx=batch_idx)
        self.intra_mesh_transformer_3 = IntraMeshTransformer(self.gaussian_dim, num_heads=4)
        self.inter_mesh_transformer_3 = InterMeshTransformer(self.gaussian_dim, num_heads=4, connectivity_graph=connectivity_graph, connectivity_mask=connectivity_mask, batch_idx=batch_idx)
        self.intra_mesh_transformer_4 = IntraMeshTransformer(self.gaussian_dim, num_heads=4)
        self.inter_mesh_transformer_4 = InterMeshTransformer(self.gaussian_dim, num_heads=4, connectivity_graph=connectivity_graph, connectivity_mask=connectivity_mask, batch_idx=batch_idx)
        self.intra_mesh_transformer_5 = IntraMeshTransformer(self.gaussian_dim, num_heads=4)
        self.inter_mesh_transformer_5 = InterMeshTransformer(self.gaussian_dim, num_heads=4, connectivity_graph=connectivity_graph, connectivity_mask=connectivity_mask, batch_idx=batch_idx)
        self.intra_mesh_transformer_6 = IntraMeshTransformer(self.gaussian_dim, num_heads=4)
        self.inter_mesh_transformer_6 = InterMeshTransformer(self.gaussian_dim, num_heads=4, connectivity_graph=connectivity_graph, connectivity_mask=connectivity_mask, batch_idx=batch_idx)

        self.gaussian_attn = CrossAttention(self.gaussian_dim, num_heads=4)
        
        self.gaussian_pred_head = nn.Sequential(
            nn.Linear(self.gaussian_dim, self.gaussian_dim),
            nn.ReLU(),
            nn.Linear(self.gaussian_dim, self.gaussian_dim//2),
            nn.ReLU(),
            nn.Linear(self.gaussian_dim//2, 14),
        )
         
        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1) # NOTE: maybe remove it if train again

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1.5, 1.5) * 1.7
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)


    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict


    def prepare_default_rays(self, device, elevation=0):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        # Hardcoded
        # manually set the camera poses for mvhumannet
        """These are the default camera poses for the MVHUMAN dataset."""
        cam_poses= torch.tensor([[[-3.2914e-01, -2.9498e-01,  8.9702e-01,  1.9025e+00],
         [ 2.6506e-02, -9.5247e-01, -3.0349e-01, -1.5521e+00],
         [ 9.4391e-01, -7.6113e-02,  3.2131e-01,  6.0368e-01],
         [ 0.0000e+00,  0.0000e+00, -2.9802e-08,  1.0000e+00]],

        [[-9.4848e-01,  6.7344e-02, -3.0959e-01, -6.8751e-01],
         [ 3.2988e-02, -9.5085e-01, -3.0790e-01, -1.5439e+00],
         [-3.1511e-01, -3.0225e-01,  8.9964e-01,  1.8549e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 3.2914e-01,  2.8916e-01, -8.9892e-01, -1.9389e+00],
         [ 2.0033e-03, -9.5217e-01, -3.0556e-01, -1.5441e+00],
         [-9.4428e-01,  9.8769e-02, -3.1397e-01, -7.4027e-01],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 9.4592e-01, -1.0298e-01,  3.0763e-01,  6.9003e-01],
         [ 4.1476e-03, -9.4436e-01, -3.2888e-01, -1.5421e+00],
         [ 3.2438e-01,  3.1237e-01, -8.9286e-01, -1.9888e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]])

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)


        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings
    
    def save_to_ply(self,points, filename):
        """
        Save 3D points to a PLY file.
        
        Args:
            points (numpy.ndarray or torch.Tensor): Array of shape [n, 3] containing 3D points.
            filename (str): The output PLY file path.
        """
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        
        num_points = points.shape[0]
        
        with open(filename, 'w') as file:
            # Write the PLY header
            file.write("ply\n")
            file.write("format ascii 1.0\n")
            file.write(f"element vertex {num_points}\n")
            file.write("property float x\n")
            file.write("property float y\n")
            file.write("property float z\n")
            file.write("end_header\n")
            
            # Write the points
            for point in points:
                file.write(f"{point[0]} {point[1]} {point[2]}\n")

    def project_points(self, points_3d, view_proj):
        """
        Project 3D points to 2D using the combined view and projection matrix.
        
        Args:
            points_3d (torch.Tensor): 3D points [bs, views, a, b, 3].
            view_proj (torch.Tensor): Combined view and projection matrix [bs, views, 4, 4].
        
        Returns:
            torch.Tensor: Projected 2D points [bs, views, a, b, 2].
        """
        bs, views, a, b, _ = points_3d.shape

        # Add a homogeneous coordinate to the 3D points
        ones = torch.ones((bs, views, a, b, 1), device=points_3d.device)
        points_3d_h = torch.cat([points_3d, ones], dim=-1)  # [bs, views, a, b, 4]

        view_proj = view_proj[:,:views,...]
        # Project the points to 2D using the combined view and projection matrix
        points_proj = torch.einsum('bvhwk,bvkl->bvhwl', points_3d_h, view_proj)  # [bs, views, a, b, 4]

        # Normalize the projected points to get the final 2D coordinates
        points_2d = points_proj[..., :2] / points_proj[..., 3:4]  # [bs, views, a, b, 2]

        return points_2d
    
    def visualize_projected_points(self, projected_points, image_size=(640, 480)):
        from matplotlib import pyplot as plt
        """
        Visualize the projected 2D points on an image using matplotlib.
        
        Args:
            projected_points: Tensor of shape [bs, views, N, 2] containing the 2D projected points.
            image_size: Tuple (width, height) representing the size of the image.
        """
        bs, views, N, _ = projected_points.shape
        
        for batch in range(bs):
            for view in range(views):
                # Create a blank white image
                image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
                
                # Get the 2D points for this batch and view
                points_2d = projected_points[batch, view].cpu().numpy()
                
                # Plot the points on the image
                plt.figure(figsize=(10, 6))
                plt.imshow(image)
                plt.scatter(points_2d[:, 0], points_2d[:, 1], c='red', s=10)  # Plot points in red
                plt.title(f'Batch {batch}, View {view}')
                plt.axis('off')  # Hide axes
                plt.savefig(f'projected_points_batch_{batch}_view_{view}.png')
    
    def crop(self, img, length=512):
        bs, views, ch, h, w = img.shape
        if h > w:
            start = (h - w) // 2
            img_cropped = img[:, :, :, start:start + w, :]
        else:
            start = (w - h) // 2
            img_cropped = img[:, :, :, :, start:start + h]

        # Reshape to [N, C, H, W] format
        img_cropped = img_cropped.view(-1, ch, img_cropped.shape[-2], img_cropped.shape[-1])

        # Resize the image using torch.nn.functional.interpolate
        img_resized = F.interpolate(img_cropped, size=(length, length), mode='bilinear', align_corners=False)

        # Reshape back to [bs, views, channel, length, length]
        img_resized = img_resized.view(bs, views, ch, length, length)

        return img_resized
    

    def forward_gaussians(self, images):
        # images: [B, 4, 9, H, W] 256
        # face_indices: [B, 4, H', W'] 512
        # return: Gaussians: [B, dim_t]

        B, V, C, H, W = images.shape
        # Hi, Wi = face_indices.shape[-2:]
        # H and W are 256, 256
        device = next(self.unet.parameters()).device
        images = images.view(B*V, C, H, W).to(device)
        x, gaussian = self.unet(images) # [B*4, 14, h, w] # already 64*64 now
        
        
        x = x.reshape(B, 4, -1, self.opt.splat_size, self.opt.splat_size) # [B, 4, 14, h, w]
        # in this setting, Gaussian maps are 64*64, with 14 channels. We can modify it to (64,64) but 128 channels
        x = x.permute(0, 1, 3, 4, 2) # [B, 4, h, w, 128]

        features = x 
        
        gaussian = self.conv(gaussian) # [B*4, 14, h, w]
        gaussian = gaussian.reshape(B, 4, -1, self.opt.splat_size, self.opt.splat_size)
        gaussian = gaussian.permute(0, 1, 3, 4, 2) 
        
        raw_gaussian = gaussian.clone()
        
        pos = self.pos_act(gaussian[..., 0:3]) # torch.Size([1, 4, 64, 64, 3])
        #  
        # pos = self.pos_act(x[..., 0:3]) / 10.0 # [B, N, 3]
        opacity = self.opacity_act(gaussian[..., 3:4])
        scale = self.scale_act(gaussian[..., 4:7])
        rotation = self.rot_act(gaussian[..., 7:11])
        rgbs = self.rgb_act(gaussian[..., 11:])

        gaussian = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)
        #  
        
        # torch.Size([1, 4, 64, 64, 65])
        return features ,gaussian, raw_gaussian

    def knn_torch_cluster(self, pred_points, smpl_points):
        """
        Finds the nearest SMPL vertex (from smpl_points) for each predicted point.
        Uses torch_cluster.knn which runs on GPU.

        Args:
            pred_points (torch.Tensor): [N, 3]
            smpl_points (torch.Tensor): [M, 3]

        Returns:
            nearest_indices (torch.Tensor): [N] indices of nearest smpl_points for each pred_point.
        """
        # Make sure both tensors are contiguous and on the same GPU device.
        pred_points = pred_points.contiguous()
        smpl_points = smpl_points.contiguous()
        edge_index = torch_cluster.knn(smpl_points, pred_points, k=1)
        # edge_index[0] contains indices in smpl_points for each pred_point.
        return edge_index[1]
    
    def forward(self, video_data, extra_poses=None):
        # data: output of the dataloader
        # return: loss
        
        frame_ids = list(video_data.keys())
        # frame_ids.sort()
        
        results = {}
        results["gaussians"]=[]
        loss = 0

        device = next(self.unet.parameters()).device

        all_frame_batch_grouped_tensors = []
        all_frame_batch_grouped_masks = []

        for frame_id in frame_ids:

            data = video_data[frame_id]
            images = data['input'].to(device) # [B, 4, 9, h, W], input features
            # face_indices = data['face_indices'] 
            """images cannot be BF16, convert it to F16"""
            camera_extrinsics = data["cam_view"].transpose(2,3)[:,:images.shape[1],:,:]

            # use the first view to predict gaussians
            features, the_gaussian, raw_gaussian = self.forward_gaussians(images) # [B, N, 14]
            
            # turn on this switch if you want to train LGM 
            TRAIN_LGM = False
            if TRAIN_LGM:
                bg_color = torch.ones(3, dtype=torch.float32, device=the_gaussian.device)
                the_render_gaussian = the_gaussian.clone().view(features.shape[0], -1, 14) # opacity:3rd
                # 65536
                # opacity > 0.02 : 9415
                # opacity > 0.1 : 8983 
                opa_cand = 9000
                dist_loss_k = 0.4
                selected_render_gaussian = []
                for i in range(features.shape[0]):
                    cur_render_gaussian = the_render_gaussian[i]
                    opacities = cur_render_gaussian[..., 3]
                    sorted_values, sorted_indices = torch.sort(opacities, descending=True)
                    top_indices = sorted_indices[:9000]
                    cur_render_gaussian = cur_render_gaussian[top_indices]
                    selected_render_gaussian.append(cur_render_gaussian)
                    
                the_render_gaussian = torch.stack(selected_render_gaussian, dim=0)
                     
                smpl_vertices = data['transformed_smpl_vertices']
                # the smpl_vertices and the_render_gaussian are very aligned
                the_pred_points = the_render_gaussian[..., 0:3] # bs, 9000, 3
                
                # you can comment this part if you don't want to save the points
                for this_bs in range(the_pred_points.shape[0]):
                    this_points = the_pred_points[this_bs]
                    this_points = this_points.detach()
                    root_dir = data['root_dir'][this_bs]
                    human = data['human'][this_bs]
                    pose = data['pose'][this_bs]
                    big_dir = os.path.join(root_dir, human, "save_pred_points")
                    os.makedirs(big_dir, exist_ok=True)
                    save_dir = os.path.join(big_dir, f'{pose}.ply')
                    self.save_to_ply(this_points, save_dir)
                
                # calculate the distance between the predicted points and the smpl vertices
                # for each point in pred
                # [bs, 9000]
                min_dists = torch.cdist(the_pred_points, smpl_vertices).min(dim=2).values * 10
                min_dist_loss = (dist_loss_k * (min_dists ** 3)).mean()*features.shape[0]
                
                loss = loss + min_dist_loss


                # use the other views for rendering and supervision
                the_render_results = self.gs.render(the_render_gaussian, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
                pred_images = the_render_results['image'] # [B, 8, 3, H', W']
                pred_alphas = the_render_results['alpha'] # [B, 8, 1, H', W']
                
                # crop to 512 and do the supervision
                gt_images = data['images_output'] # [B, 8, 3, h, w]
                gt_masks = data['masks_output'] # [B, 8, 1, h, w]
                
                target_h, target_w = gt_images.shape[-2:]
                assert target_h == target_w, "The target image should be square"
                pred_images = self.crop(pred_images, target_h)
                pred_alphas = self.crop(pred_alphas, target_h)
                
                results['images_pred'] = pred_images
                gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

                loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
                loss = loss + loss_mse

                
                
                if self.opt.lambda_lpips > 0:
                    loss_lpips = self.lpips_loss(
                        # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                        # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                        # downsampled to at most 256 to reduce memory cost
                        F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                        F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                    ).mean()
                    results['loss_lpips'] = (results['loss_lpips'] + loss_lpips) if 'loss_lpips' in results else loss_lpips
                    loss = loss + self.opt.lambda_lpips * loss_lpips
                    
                    
                """Optional: Visualization of rendered images"""
                # save_render = '2d_visual/render_init_mvhuman_small'
                # os.makedirs(save_render, exist_ok=True)
                # from PIL import Image
                # for i in range(pred_images.shape[1]):
                #     img = pred_images[0, i]  # Extract the i-th image
                #     save_path = os.path.join(save_render, f'image_frame_{frame_id}view_{i}.png')
                #     img_np = img.permute(1, 2, 0).detach().cpu().numpy()  # Convert to NumPy array and change to HWC format
                #     img_np = (img_np * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
                #     #  
                #     img_pil = Image.fromarray(img_np)
                #     img_pil = img_pil.resize((1224, 1024))
                #     img_pil.save(save_path)
                #     rendered_rgb.append(img_pil)
                # #  
            
            else: # not train LGM but train mesh graph
                
                
                bg_color = torch.ones(3, dtype=torch.float32, device=the_gaussian.device)
                cat_gaussian = torch.cat([the_gaussian, features, raw_gaussian], dim=-1) # [B, 4, h, w, 14+128+14]
                the_feature_gaussian = cat_gaussian.clone().view(features.shape[0], -1, 14+features.shape[-1]+14) # opacity:3rd
                # 65536
                # opacity > 0.02 : 9415
                # opacity > 0.1 : 8983 
                opa_cand = 9000
                dist_loss_k = 0.4
                selected_feature_gaussian = []
                for i in range(features.shape[0]):
                    cur_render_gaussian = the_feature_gaussian[i]
                    opacities = cur_render_gaussian[..., 3]
                    sorted_values, sorted_indices = torch.sort(opacities, descending=True)
                    top_indices = sorted_indices[:opa_cand]
                    cur_render_gaussian = cur_render_gaussian[top_indices]
                    selected_feature_gaussian.append(cur_render_gaussian)
                
                the_feature_gaussian = torch.stack(selected_feature_gaussian, dim=0) # [1,9000,42]
                
                smpl_vertices = torch.tensor(data['optimized_vertices'], dtype=the_feature_gaussian.dtype, device=the_feature_gaussian.device)
                # the smpl_vertices and the_render_gaussian are very aligned
                the_pred_points = the_feature_gaussian[..., 0:3]
                batch_grouped_tensors = []
                batch_grouped_masks = []
                for b in range(the_feature_gaussian.shape[0]):
                    # Get predicted points and SMPL vertices for the current batch item.
                    pred_points = the_feature_gaussian[b, :, 0:3]  # [num_gaussians, 3]
                    curr_smpl_vertices = smpl_vertices[b]           # [num_vertices, 3]
                    
                    nearest_indices = self.knn_torch_cluster(pred_points, curr_smpl_vertices)  # [num_gaussians]
                    # Sort nearest_indices and corresponding gaussians on GPU
                    sorted_values, order = torch.sort(nearest_indices)
                    sorted_gaussians = the_feature_gaussian[b][order]

                    # Get unique vertex indices and counts
                    unique_vertices, counts = torch.unique_consecutive(sorted_values, return_counts=True)
                    # Use torch.split to slice sorted_gaussians according to counts
                    # split_gaussians = torch.split(sorted_gaussians, counts.tolist())

                    num_vertices = self.vertices
                    gaussian_dim = sorted_gaussians.shape[1]
                    max_gaussians = self.max_gaussian_per_frame_per_vertex  # e.g. 10

                    # Preallocate output tensor and mask.
                    grouped_tensor = torch.zeros(num_vertices, max_gaussians, gaussian_dim, device=sorted_gaussians.device)
                    group_mask = torch.ones(num_vertices, max_gaussians, dtype=torch.bool, device=sorted_gaussians.device)

                    # Compute counts per vertex (already a tensor on GPU)
                    unique_vertices, counts = torch.unique_consecutive(sorted_values, return_counts=True)

                    # Create a [num_groups, max_group_count] index matrix
                    max_count = counts.max()
                    idx_matrix = torch.arange(max_count, device=counts.device).unsqueeze(0).expand(counts.shape[0], max_count)
                    # Mask out entries that exceed the group count for each vertex.
                    local_idx_full = idx_matrix[idx_matrix < counts.unsqueeze(1)]
                    
                    
                    # Create a mask to filter only indices less than max_gaussians.
                    valid_mask = local_idx_full < max_gaussians
                    filtered_local_idx = local_idx_full[valid_mask]

                    # Also filter the corresponding sorted_values and sorted_gaussians.
                    filtered_sorted_values = sorted_values[valid_mask]
                    filtered_sorted_gaussians = sorted_gaussians[valid_mask]

                    # Use advanced indexing to scatter the valid sorted gaussians into grouped_tensor.
                    grouped_tensor[filtered_sorted_values, filtered_local_idx, :] = filtered_sorted_gaussians
                    group_mask[filtered_sorted_values, filtered_local_idx] = False
                        
                    batch_grouped_tensors.append(grouped_tensor)
                    batch_grouped_masks.append(group_mask)
                
                batch_grouped_tensors = torch.stack(batch_grouped_tensors, dim=0) # torch.Size([bs, 6890, 10, 156])
                batch_grouped_masks = torch.stack(batch_grouped_masks, dim=0) # torch.Size([bs, 6890, 10])
                #  
                all_frame_batch_grouped_tensors.append(batch_grouped_tensors)
                all_frame_batch_grouped_masks.append(batch_grouped_masks)
        
        if TRAIN_LGM:
            results['loss'] = loss
            
            with torch.no_grad():
                psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
                results['psnr'] = psnr
            
            return results
    
        del data
        
        all_frame_batch_grouped_tensors = torch.stack(all_frame_batch_grouped_tensors, dim=0) # torch.Size([5, 1, 6890, 10, 156])
        all_frame_batch_grouped_masks = torch.stack(all_frame_batch_grouped_masks, dim=0) # torch.Size([5, 1, 6890, 10])
        all_frame_batch_grouped_tensors = all_frame_batch_grouped_tensors.permute(1, 0, 2, 3, 4) # torch.Size([1, 5, 6890, 10, 156])
        all_frame_batch_grouped_masks = all_frame_batch_grouped_masks.permute(1, 0, 2, 3) # torch.Size([1, 5, 6890, 10])
        def soften_mask(mask):
            """
            GPU version.
            Input: mask tensor of shape [6890, 50] on GPU.
            For each row where all entries are True, set the first entry to False.
            """
            # Identify the rows where all elements are True.
            rows_all_true = mask.all(dim=1)
            # Set the first element in each such row to False.
            mask[rows_all_true, 0] = False
            return mask
        
        for bs in range(self.opt.batch_size):
            # start = time.time()
            
            # this part is the core of our alg.
            
            # gather all the relevant info here: Gaussian attributes and features
            features_this_batch = all_frame_batch_grouped_tensors[bs] # [5, 6890, 10, 156]
            
            # the precomputed masks
            masks_this_batch = all_frame_batch_grouped_masks[bs] # [5, 6890, 10]
            big_tensor = features_this_batch.permute(1, 0,2, 3).reshape(self.vertices, -1, features_this_batch.shape[-1]) # [6890, 50, 156]
            attn_mask = masks_this_batch.permute(1, 0, 2).reshape(self.vertices, -1) # [6890, 50]
            attn_mask = soften_mask(attn_mask)
            
            ############################################################################################
            """# do intra mesh attn"""
            mesh_features = big_tensor[..., 14:14+128].permute(1,0,2) # [M, 6890, 128]
            
            mesh_queries = self.intra_mesh_transformer_1(self.mesh_queries.permute(1,0,2),mesh_features, attn_mask) # batch second # has residue in it
            mesh_queries = self.inter_mesh_transformer_1(mesh_queries) # [5, 6890, 128]
            
            mesh_queries = self.intra_mesh_transformer_2(mesh_queries,mesh_features, attn_mask) 
            mesh_queries = self.inter_mesh_transformer_2(mesh_queries) # [5, 6890, 128]

            mesh_queries = self.intra_mesh_transformer_3(mesh_queries,mesh_features, attn_mask) 
            mesh_queries = self.inter_mesh_transformer_3(mesh_queries) # [5, 6890, 128]
            #  
            mesh_queries = self.intra_mesh_transformer_4(mesh_queries,mesh_features, attn_mask) 
            mesh_queries = self.inter_mesh_transformer_4(mesh_queries) # [5, 6890, 128]
            
            mesh_queries = self.intra_mesh_transformer_5(mesh_queries,mesh_features, attn_mask)
            mesh_queries = self.inter_mesh_transformer_5(mesh_queries) # [5, 6890, 128]
            
            mesh_queries = self.intra_mesh_transformer_6(mesh_queries,mesh_features, attn_mask) 
            mesh_queries = self.inter_mesh_transformer_6(mesh_queries)  
            
            """# do final Gaussian-Mesh attn"""
            cur_frame_big_tensor = features_this_batch[0] # [6890, 10, 156]
            
            strict_mask = masks_this_batch[0] # [6890, 10]
            
            cur_frame_mesh_features = cur_frame_big_tensor[..., 14:14+128].permute(1,0,2) # [M, 6890, 128]
            cur_frame_renewed_features = self.gaussian_attn(cur_frame_mesh_features, mesh_queries, mesh_queries) # [M, 6890, 128]
            
            cur_frame_renewed_features = cur_frame_renewed_features.permute(1,0,2) # [6890, M, 128]
            cur_valid_features = cur_frame_renewed_features[~strict_mask] 
            
            """Note: the current frame here has been changed to the zeroth frame, the name is not correct"""
            cur_frame_original_gaussian = cur_frame_big_tensor[..., 14+128:] # [6890, M, 14]
            #  
            cur_frame_original_gaussian = cur_frame_original_gaussian[~strict_mask]
            selected = torch.nonzero(~strict_mask)[:,0] # indices of gaussians to mesh
            
            # the out layer in unet
            cur_frame_gaussian = self.unet.norm_out(cur_valid_features) # [13187, 14]
            cur_frame_gaussian = F.silu(cur_frame_gaussian)

            
            gaussian_mod = self.gaussian_pred_head(cur_frame_gaussian) # [13187, 14]
            gaussian = cur_frame_original_gaussian + gaussian_mod * 0.5 # residue
            
            pos = self.pos_act(gaussian[..., 0:3]) 

            """Optional: Save the point cloud to a PLY file"""
            # output_path = './geo_viz/added_pc.ply'
            # with open(output_path, 'w') as f:
            #     # Write header.
            #     f.write("ply\n")
            #     f.write("format ascii 1.0\n")
            #     f.write(f"element vertex {num_points}\n")
            #     f.write("property float x\n")
            #     f.write("property float y\n")
            #     f.write("property float z\n")
            #     f.write("end_header\n")
                
            #     # Write each point.
            #     for x, y, z in points_np:
            #         f.write(f"{x} {y} {z}\n")

            # print(f"Saved point cloud to {output_path}")

            # pos = self.pos_act(x[..., 0:3]) / 10.0 # [B, N, 3]
            opacity = self.opacity_act(gaussian[..., 3:4])
            scale = self.scale_act(gaussian[..., 4:7])
            rotation = self.rot_act(gaussian[..., 7:11])
            rgbs = self.rgb_act(gaussian[..., 11:])

            final_gaussian = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)
            
            device = the_gaussian.device
            
            # get smpl params
            poses = video_data[frame_ids[0]]["optimized_poses"][bs].to(device)
            betas = video_data[frame_ids[0]]["optimized_betas"][bs].to(device)
            transl = video_data[frame_ids[0]]["optimized_transl"][bs].to(device)
            scale = video_data[frame_ids[0]]["optimized_scale"][bs].to(device)
            
            
            global_orient=poses[:, :3]
            
            # get rotation matrix
            output, T = self.smpl_model(betas=betas.to(device), global_orient=global_orient.to(device),
                        body_pose=poses[:, 3:].to(device)) # T: [1, 6890, 4, 4]
            
            T = T[0].float() # [6890,4,4]
            
            # final_gaussian [7628,14]
            # T [6890,4,4]
            # select [7628], ranging from 0 to 6889
            T_for_gaussian = T[selected,:,:]
            gaussian_xyz_opengl = final_gaussian[..., 0:3]
            smpl_vertices_opengl = gaussian_xyz_opengl
             
            smpl_vertices = smpl_vertices_opengl.clone().to(device)
            smpl_vertices[...,1] = -smpl_vertices_opengl[...,2]
            smpl_vertices[...,2] = smpl_vertices_opengl[...,1]

            
            smpl_vertices_trans = (smpl_vertices-transl) / scale
            
            ones = torch.ones((smpl_vertices_trans.shape[0], 1), device=smpl_vertices_trans.device, dtype=torch.float32)
            
            vertices_h = torch.cat([smpl_vertices_trans, ones], dim=1)  # shape: (6890, 4)

            # Compute the inverse of T for each vertex.
            T_inv = torch.inverse(T_for_gaussian)  # shape: (6890, 4, 4)

            # Apply the inverse transformation.
            # Each vertex's homogeneous coordinate is multiplied by its corresponding T_inv.
            vertices_pre_h = torch.bmm(T_inv, vertices_h.unsqueeze(-1)).squeeze(-1)  # shape: (6890, 4)

            # Convert back to 3D coordinates.
            vertices_pre = vertices_pre_h[:, :3] / vertices_pre_h[:, 3].unsqueeze(1)
            
            
            """Get the canonical vertices from T_inv"""
            gaussians_xyz_in_canonical = vertices_pre # without translation and scale
            
            
            gaussian_for_diff_frames = []
            # gaussian_for_diff_frames.append(final_gaussian) # gaussian for the first frame
            
            
            """Deform the other frames from canonical"""
            for i in range(0, len(frame_ids)):
                poses = video_data[frame_ids[i]]["optimized_poses"][bs].to(device)
                betas = video_data[frame_ids[i]]["optimized_betas"][bs].to(device)
                transl = video_data[frame_ids[i]]["optimized_transl"][bs].to(device)
                scale = video_data[frame_ids[i]]["optimized_scale"][bs].to(device)
                
                # get the T from the smpl model with this frame's pose and beta
                global_orient=poses[:, :3]
                # we only want T
                output, T = self.smpl_model(betas=betas.to(device), global_orient=global_orient.to(device),
                            body_pose=poses[:, 3:].to(device)) # T: [1, 6890, 4, 4]
                T = T[0].float() # [6890,4,4]
                T_for_gaussian = T[selected,:,:]
                gaussian_canonical = gaussians_xyz_in_canonical.clone()
                
                # apply the T to the canonical gaussians
                ones = torch.ones((gaussian_canonical.shape[0], 1), device=gaussian_canonical.device, dtype=torch.float32)
                vertices_h = torch.cat([gaussian_canonical, ones], dim=1)  # shape: (6890, 4)
                vertices_pre_h = torch.bmm(T_for_gaussian, vertices_h.unsqueeze(-1)).squeeze(-1)
                vertices_pre = vertices_pre_h[:, :3] / vertices_pre_h[:, 3].unsqueeze(1)
                
                vertices_pre = vertices_pre * scale + transl

                # convert back to opengl
                vertices_pre_opengl = vertices_pre.clone()
                vertices_pre_opengl[...,1] = vertices_pre[...,2]
                vertices_pre_opengl[...,2] = -vertices_pre[...,1]
                
                gaussians_this_frame = final_gaussian.clone()
                gaussians_this_frame[..., 0:3] = vertices_pre_opengl
                gaussian_for_diff_frames.append(gaussians_this_frame)
            
            
            gaussian_for_diff_frames = torch.stack(gaussian_for_diff_frames, dim=0) # [8, 13187, 14]    
            results["gaussians"].append(gaussian_for_diff_frames)       
            
            if extra_poses is None:
                return results
            
            
            """Optional"""
            """Those without supervision: the novel pose gaussians"""
            extra_gaussian_for_poses = []
            frame_ids = list(extra_poses.keys())
            for i in range(0, len(frame_ids)):
                poses = extra_poses[frame_ids[i]]["optimized_poses"][bs].to(device)
                betas = extra_poses[frame_ids[i]]["optimized_betas"][bs].to(device)
                transl = extra_poses[frame_ids[i]]["optimized_transl"][bs].to(device)
                scale = extra_poses[frame_ids[i]]["optimized_scale"][bs].to(device)
                
                # get the T from the smpl model with this frame's pose and beta
                global_orient=poses[:, :3]
                output, T = self.smpl_model(betas=betas.to(device), global_orient=global_orient.to(device),
                            body_pose=poses[:, 3:].to(device)) # T: [1, 6890, 4, 4]
                T = T[0].float() # [6890,4,4]
                T_for_gaussian = T[selected,:,:]
                gaussian_canonical = gaussians_xyz_in_canonical.clone()
                
                # apply the T to the canonical gaussians
                ones = torch.ones((gaussian_canonical.shape[0], 1), device=gaussian_canonical.device, dtype=torch.float32)
                vertices_h = torch.cat([gaussian_canonical, ones], dim=1)  # shape: (6890, 4)
                vertices_pre_h = torch.bmm(T_for_gaussian, vertices_h.unsqueeze(-1)).squeeze(-1)
                vertices_pre = vertices_pre_h[:, :3] / vertices_pre_h[:, 3].unsqueeze(1)
                
                vertices_pre = vertices_pre * scale + transl

                # convert back to opengl
                vertices_pre_opengl = vertices_pre.clone()
                vertices_pre_opengl[...,1] = vertices_pre[...,2]
                vertices_pre_opengl[...,2] = -vertices_pre[...,1]
                
                gaussians_this_frame = final_gaussian.clone()
                gaussians_this_frame[..., 0:3] = vertices_pre_opengl
                extra_gaussian_for_poses.append(gaussians_this_frame)
            
            results["novel_pose_gaussians"]=extra_gaussian_for_poses

        
        return results

    
    def get_transformation_matrices(self, pose_params, model_path = './smpl_renderer/mvhuman_tools/visual_smpl/smpl_', device='cpu'):
        """
        Given pose parameters (e.g. global_orient and body_pose in axis-angle format),
        load the SMPLX model, perform a forward pass, and return the transformation matrices.
        
        Parameters:
            pose_params (dict): Dictionary with keys:
                - 'global_orient': tensor of shape (1, 3)
                - 'body_pose': tensor of shape (1, 69)  # example: 23 joints * 3 axis-angle values
            model_path (str): Path to the SMPLX model files.
            device (str): Device to run the model on.
        
        Returns:
            dict: A dictionary of transformation matrices, for example:
                - 'global_orient': tensor of shape (1, 3, 3)
                - 'joints_transforms': list of tensors, each of shape (4, 4)
        """
        # Create a SMPLX model instance.
        model = smplx.create(model_path=model_path, model_type='smpl', gender='neutral', ext='pkl', device=device)
        model= model.to(device)
        
        
        # Forward the pose parameters.
        global_orient=pose_params[:, :3]
        if global_orient.shape[-1] == 3:  # axis-angle representation
            R = axis_angle_to_matrix(global_orient)  # returns shape (1, 3, 3)
        else:
            R = global_orient  # already rotation matrix
        output = model(global_orient=global_orient.to(device),
                    body_pose=pose_params[:, 3:].to(device))
        #  
        # Get the global rotation matrix.
        global_orient = output.global_orient  # (1, 3, 3)
        
        # You can extract full joint transformations from the output. For example, each joint has a 4x4 transformation.
        joints_transforms = output.get('joints_transform', None)  # This depends on your SMPLX version.
        
        if joints_transforms is None:
            # Alternatively, you can compute them from the joint rotation matrices and translations.
            joints = output.joints  # (1, J, 3)
            # For each joint, build a 4x4 matrix. (This is an example; check your SMPLX API.)
            joints_transforms = []
            for j in range(joints.shape[1]):
                R = output.joint_rotations[:, j]  # (1, 3, 3)
                t = joints[:, j].unsqueeze(-1)      # (1, 3, 1)
                transform = torch.cat([torch.cat([R, t], dim=-1),
                                    torch.tensor([0,0,0,1], device=device).view(1,1,4)], dim=1)
                joints_transforms.append(transform.squeeze(0))
        
        return {
            'global_orient': global_orient,
            'joints_transforms': joints_transforms,
        }
                
                
    def recover_canonical_gaussians(self, gaussians, pose_params, translation, scale, model_path='./smpl_renderer/mvhuman_tools/visual_smpl/smpl_', device='cpu'):
        """
        Recover gaussians in the canonical (standing) pose from the posed gaussians.
        
        Parameters:
            gaussians (torch.Tensor): Tensor of shape [N, 14] where:
                - columns 0:3 are positions,
                - columns 7:11 are quaternions (w, x, y, z).
            pose_params (dict): SMPL pose parameters with keys:
                - 'global_orient': tensor of shape (1, 3)
                - 'body_pose': tensor of shape (1, 69)
            translation (torch.Tensor): Global translation applied to the gaussians (shape [3] or broadcastable).
            scale (float or torch.Tensor): Global scale applied to the gaussians.
            model_path (str): Path to the SMPLX model files.
            device (str): Device to run on.
            
        Returns:
            torch.Tensor: Gaussians in canonical pose with the same shape as input.
        """
        # Retrieve SMPLX transformation matrices.
        #  
        
        trans_dict = self.get_transformation_matrices(pose_params, model_path=model_path, device=device)
        
        # Invert the global orientation for position and rotation.
        R = trans_dict['global_orient'][0]  # shape: (3, 3)
        R_inv = R.transpose(0, 1)  # inverse rotation
        
        canonical_gaussians = gaussians.clone()
        
        # Recover canonical positions.
        # First, undo translation and scale:
        pos = gaussians[..., 0:3]
        pos_transformed = (pos - translation) / scale
        canonical_pos = torch.matmul(pos_transformed, R_inv)
        canonical_gaussians[..., 0:3] = canonical_pos
        
        # Recover canonical rotation.
        # Assume gaussian[..., 7:11] contains quaternion in (w, x, y, z) order.
        gaussian_quat = gaussians[..., 7:11]
        # Convert SMPL global orientation (matrix) to quaternion.
        global_quat = matrix_to_quaternion(R.unsqueeze(0)).squeeze(0)  # shape: (4,)
        global_quat_inv = quaternion_invert(global_quat)
        # Expand global_quat_inv to match dimensions.
        global_quat_inv = global_quat_inv.unsqueeze(0).expand_as(gaussian_quat)
        
        canonical_quat = quaternion_multiply(global_quat_inv, gaussian_quat)
        canonical_gaussians[..., 7:11] = canonical_quat

        return canonical_gaussians

    def compute_world_to_photo_matrix(self, w2c: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Combine the world-to-camera (w2c) 4x4 matrix with a 3x3 intrinsic matrix
        to form a full 4x4 projection that maps world coordinates to pixel coordinates.

        Args:
            w2c:         [bs, views, 4, 4] world-to-camera transform
            intrinsics:  [bs, views, 3, 3] camera intrinsic matrix

        Returns:
            proj_matrix: [bs, views, 4, 4] world-to-pixel transform
        """
        # Copy shape info
        batch_dims = w2c.shape[:-2]  # e.g. [bs, views]
        
        # Create a 4x4 from the 3x3 intrinsic
        # Start with an identity, then fill in the top-left 3x3 for the focal & principal point
        K_4x4 = torch.eye(4, device=w2c.device, dtype=w2c.dtype)
        K_4x4 = K_4x4.unsqueeze(0).expand(*batch_dims, 4, 4).clone()  # match batch shape
        K_4x4[..., :3, :3] = intrinsics  # embed the 3x3 intrinsics in the top-left
        
        # The final projection is K * [R|t], i.e. K_4x4 @ w2c
        proj_matrix = K_4x4 @ w2c
        return proj_matrix
