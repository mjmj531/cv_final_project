import torch
import numpy as np
import cv2
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import os

# 镜像访问huggingface
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# model = VGGT()
# _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
# model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

def extract_foreground(rgb_image, mask):
    """使用掩码提取前景(人体)，去除背景"""
    # 确保掩码是二值图像
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # 二值化处理，确保掩码为0和255
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 创建只有前景的图像
    foreground = np.zeros_like(rgb_image)
    foreground[binary_mask > 0] = rgb_image[binary_mask > 0]
    
    return foreground

# Load and preprocess example images (replace with your own image paths)
# 文件夹路径
folder_path = "dataset/data1-humanbody1"

# 获取所有RGB图像和掩码图像的路径
rgb_images = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith("rgb_") and not f.startswith("rgb_foreground_") and f.endswith(".png")])
mask_images = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith("msk_") and f.endswith(".png")])

# 确保RGB图像和掩码图像数量一致
assert len(rgb_images) == len(mask_images), "RGB images and mask images count do not match."

# 加载图像并提取前景
foreground_images = [extract_foreground(cv2.imread(rgb_image), cv2.imread(mask_image)) for rgb_image, mask_image in zip(rgb_images, mask_images)]

# 保存前景图像到临时文件夹
foreground_image_paths = []
for i, foreground_image in enumerate(foreground_images):
    temp_path = os.path.join('dataset/data1-humanbody1', f"foreground_{i}.png")
    cv2.imwrite(temp_path, foreground_image)
    foreground_image_paths.append(temp_path)

# 加载并预处理图像
images = load_and_preprocess_images(foreground_image_paths).to(device)
# breakpoint()

# 指定保存文件夹路径
output_folder = "dataset/data1-humanbody1/vggt_output"
os.makedirs(output_folder, exist_ok=True)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)
                
    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    print("extrinsic shape:", extrinsic.shape)
    print("intrinsic shape:", intrinsic.shape)
    # print("extrinsic:", extrinsic)
    # print("intrinsic:", intrinsic)

    # 保存相机内外参数
    # extrinsic: [1, 16, 3, 4], intrinsic: [1, 16, 3, 3]
    extrinsic = extrinsic.squeeze(0)  # [16, 3, 4]
    intrinsic = intrinsic.squeeze(0)  # [16, 3, 3]
    assert extrinsic.shape[0] == len(foreground_image_paths)
    assert intrinsic.shape[0] == len(foreground_image_paths)

    for i in range(len(foreground_image_paths)):
        extrinsic_path = os.path.join(output_folder, f"extrinsics_{i}.npy")
        intrinsic_path = os.path.join(output_folder, f"intrinsics_{i}.npy")
        np.save(extrinsic_path, extrinsic[i].cpu().numpy())
        np.save(intrinsic_path, intrinsic[i].cpu().numpy())

    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    # Predict Point Maps
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
    print("point_map shape:", point_map.shape)  
        
    # Construct 3D Points from Depth Maps and Cameras
    # which usually leads to more accurate 3D points than point map branch
    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                extrinsic.squeeze(0), 
                                                                intrinsic.squeeze(0))
    print("point_map_by_unprojection shape:", point_map_by_unprojection.shape)

    # Predict Tracks
    # choose your own points to track, with shape (N, 2) for one scene
    query_points = torch.FloatTensor([[100.0, 200.0], 
                                        [60.72, 259.94]]).to(device)
    track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])