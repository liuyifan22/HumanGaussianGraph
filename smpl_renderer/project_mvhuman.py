from mvhuman_tools.visual_smpl.mytools.camera_utils import read_camera_mvhumannet
from mvhuman_tools.visual_smpl.mytools.writer import FileWriter
import trimesh
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pickle
import tqdm
import argparse
import glob

def count_indices_png_files(folder_path):
    # Use glob to find all files ending with "indices.png"
    indices_png_files = glob.glob(os.path.join(folder_path, '*indices.png'))
    # Return the count of such files
    return len(indices_png_files)



# Start Xvfb
os.system('Xvfb :99 -screen 0 640x480x24 &')
os.environ['DISPLAY'] = ':99'

projection_results = "projection_result.jpg"
parser = argparse.ArgumentParser(description="Process two arguments.")
parser.add_argument("--start", default="200125",type=str, help="First argument")
parser.add_argument("--end", default="200126",type=str, help="Second argument")
    
args = parser.parse_args()

root_path = r'/root/avatar/mvhumannet/mvhuman_24'
smplx_render = 'smplx_render'
smplx_render_path = os.path.join(root_path, smplx_render)
human_ids = [str(i) for i in range(int(args.start),int(args.end))]

smplx_ids = [10*i for i in range(30)]
formatted_smplx_ids = [f"{id:06d}.obj" for id in smplx_ids]
pose_id = [50*i + 5 for i in range(30)]
formatted_pose_ids = [f"{id:04d}.jpg" for id in pose_id]

img_path =  r'/root/avatar/mvhumannet/mvhuman_24/200023/images_lr/22327091/0380_img.jpg'
# obj_path = r'/root/avatar/mvhumannet/mvhuman_24/200023/smplx/smplx_mesh/000075.obj'
# intri_name = r'/root/avatar/mvhumannet/mvhuman_24/200023/camera_intrinsics.json'
# extri_name =  r'/root/avatar/mvhumannet/mvhuman_24/200023/camera_extrinsics.json'
camera_render_list = ["22327091","22327116","22236236","22327073"]
# camera_render_list = ["22327091","22327102","22327116","22236236","22327073","22327107"]
# camera_scale_fn = r'/root/avatar/mvhumannet/mvhuman_24/200023/camera_scale.pkl'
# img_path =  r'data/0380_img.jpg'
# obj_path = r'data/000075_smplx.obj'
# intri_name = r'data/camera_intrinsics.json'
# extri_name =  r'data/camera_extrinsics.json'
# camera_render_list = ["CC32871A059",]
# camera_scale_fn = r'data/camera_scale.pkl'


for human_id in tqdm.tqdm(human_ids):
    print(f'Processing {human_id}')
    cur_human_path = os.path.join(root_path, human_id)
    if not os.path.exists(cur_human_path):
        continue
    render_path = os.path.join(cur_human_path, 'render_smplx')
    if os.path.exists(render_path):
        jump_flag = True
        for cam_i in camera_render_list:
            cur_camera_path = os.path.join(cur_human_path, 'render_smplx', cam_i)
            if not os.path.exists(cur_camera_path):
                jump_flag = False
                break
            num_files = count_indices_png_files(cur_camera_path)
            if num_files < len(formatted_pose_ids):
                jump_flag = False
                break
        if jump_flag:
            print (f"Already processed {human_id}")
            # continue
            
        intri_name = os.path.join(cur_human_path,'camera_intrinsics.json')
        extri_name = os.path.join(cur_human_path,'camera_extrinsics.json')
        camera_scale_fn = os.path.join(cur_human_path, 'camera_scale.pkl')
        obj_base_path = os.path.join(cur_human_path, 'smplx/smplx_mesh')
        camera_scale = pickle.load(open(camera_scale_fn, "rb"))
        if camera_scale ==120 / 65:
            image_size = [4096, 3000]
        else:
            image_size = [1224, 1024]

        cameras_gt = read_camera_mvhumannet(intri_name, extri_name, camera_scale)
        for obj_id, pose_id in zip(formatted_smplx_ids,formatted_pose_ids):
            obj_path = os.path.join(obj_base_path, obj_id)
            
            obj_data = trimesh.load(obj_path)
            vertices = np.array(obj_data.vertices)
            faces = np.array(obj_data.faces) # (20908,3)


            render_data = {}
            assert vertices.shape[1] == 3 and len(vertices.shape) == 2, 'shape {} != (N, 3)'.format(vertices.shape)
            # pid = self.pid
            pid ,nf = 0,0
            render_data = {'vertices': vertices, 'faces': faces, 'vid': pid, 'name': 'human_'}
            render_path = os.path.join(cur_human_path, 'render_smplx')
                
            os.makedirs(render_path, exist_ok=True)
            for cam_i in camera_render_list:
                cam_i = [cam_i]
                cameras = {'K': [], 'R':[], 'T':[]}

                sub_vis = cam_i
                for key in cameras.keys():
                    cameras[key] = np.stack([cameras_gt[cam][key] for cam in sub_vis])

                cur_camera_path = os.path.join(cur_human_path, 'render_smplx', cam_i[0])
                os.makedirs(cur_camera_path, exist_ok=True)
                outname = os.path.join(cur_camera_path,pose_id)
                outname_cache = ( outname)


                config={}
                write_smpl  = FileWriter(r"/", config=config)

                data_images = cv2.imread(img_path)
                render_data_input = {"0":render_data}
                shape_ori = data_images.shape
                data_images_input = cv2.resize(data_images, [image_size[0],image_size[1]])

                smpl_img = write_smpl.vis_smpl(render_data_input, [data_images_input], cameras, outname_cache, add_back=False)
    # except:
    #     continue


        # smpl_img_re = cv2.resize(smpl_img, (image_size[0],image_size[1]), interpolation=cv2.INTER_AREA)
        # smpl_mask = np.repeat(smpl_img_re[:,:,3:],3,2)/255
        # data_save = data_images_input * (1-smpl_mask) + smpl_img_re[:,:,:3]* (smpl_mask)

        # plt.imsave(projection_results, data_save.astype("uint8"))

