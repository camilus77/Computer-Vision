'''# For CUDA 11.7 (adjust version based on your NVIDIA driver)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# For CPU only (if you don't have a CUDA-compatible GPU)
# pip install torch==2.0.1 torchvision==0.15.2

# For macOS with Apple Silicon (M1/M2)
# pip install torch==2.0.1 torchvision==0.15.2

# Install Hugging Face Transformers for DepthAnything model
pip install transformers

# Install Open3D for point cloud and mesh processing
pip install open3d

# Install other required packages
pip install opencv-python
pip install matplotlib
pip install numpy'''

######################
import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2

import open3d as o3d

import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Read in sample images
folder_path = Path("../DATA/")

num_samples = 13
selection = random.sample(os.listdir(folder_path), num_samples)

selected_images = []
for i in range(num_samples):    
    pathi = str(folder_path / selection[i])
    selected_image = cv2.imread(pathi)
    selected_image = cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB)
    selected_images.append(selected_image)

exporting = True

#load the model from Hugging Face’s model repository
checkpoints = [
    "Intel/zoedepth-nyu-kitti",
    "LiheYoung/depth-anything-large-hf",
    "jingheya/lotus-depth-g-v1-0",   
    "tencent/DepthCrafter"
    ]

checkpoint = checkpoints[1]
processor = AutoImageProcessor.from_pretrained(checkpoints[1])
model = AutoModelForDepthEstimation.from_pretrained(checkpoints[1]).to("cuda")

depth_samples = []
for i in range(num_samples):
    depth_input = processor(images=selected_images[i], return_tensors="pt").to("cuda")

    # Infer model
    with torch.no_grad():
        inference_outputs = model(**depth_input)
        output_depth = inference_outputs.predicted_depth
    
    output_depth = output_depth.squeeze().cpu().numpy()
    
    depth_samples.append([selected_images[i], output_depth])


# Visualizing Depth Maps
plt.rcParams['figure.dpi'] = 300

for i in range(num_samples):
    fig, axs = plt.subplots(2, 1)
    
    axs[0].imshow(depth_samples[i][0])
    axs[0].set_title('Depth Estimation')
    axs[1].imshow(depth_samples[i][1])
    
    plt.show()

#save for further processing
for i in range(num_samples):
    depth_image = depth_samples[i][1]
    color_image = depth_samples[i][0]
    width, height = depth_image.shape

    depth_image = (depth_image * 255 / np.max(depth_image)).astype('uint8')
    color_image = cv2.resize(color_image, (height, width))
    
    cv2.imwrite('../RESULTS/'+str(i)+'.png', cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    cv2.imwrite('../RESULTS/'+str(i)+'_depth.png', depth_image)

# prepare our depth map and color image for conversion to a point cloud
i = -4

depth_image = depth_samples[i][1]
color_image = depth_samples[i][0]
width, height = depth_image.shape

depth_image = (depth_image * 255 / np.max(depth_image)).astype('uint8')
color_image = cv2.resize(color_image, (height, width))

###create an RGBD image (a combination of color and depth) and set up our pinhole camera model
depth_o3d = o3d.geometry.Image(depth_image)
image_o3d = o3d.geometry.Image(color_image)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()


'''If focal length is too small: Objects appear "stretched" in depth
   If focal length is too large: Objects appear "compressed" in depth'''

# to create metrically accurate 3D reconstructions that match real-world scale:
fx = fy = width * 0.8  # A good approximation for a standard lens
cx, cy = width/2, height/2  # Center of the image

camera_intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

#generate and visualize our point cloud
pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
o3d.visualization.draw_geometries([pcd_raw])

# Enhancing Your Point Clouds: outliers removal
cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
pcd = pcd_raw.select_by_index(ind)

# estimate normals
pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction()
o3d.visualization.draw_geometries([pcd])

## Creating Point Clouds with Orthographic Projection
def depth_to_pointcloud_orthographic(depth_map, image, scale_factor=255):

    height, width = depth_map.shape

    # Create a grid of pixel coordinates
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Scale the depth values
    z = (depth_map / scale_factor) * height/2

    # Create 3D points (x and y are pixel coordinates, z is from the depth map)
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Filter out points with zero depth
    mask = points[:, 2] != 0
    points = points[mask]

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    
    # Add colors to the point cloud
    colors = image.reshape(-1, 3)[mask] / 255.0  # Normalize color values to [0, 1]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=1)
    inlier_cloud = pcd.select_by_index(ind)

    return inlier_cloud, z, height, width 

# Convert depth map and image to point cloud
point_cloud, z, height, width  = depth_to_pointcloud_orthographic(depth_map, image)
o3d.visualization.draw_geometries([point_cloud])

##Creating 3D Meshes from Point Clouds
point_cloud.estimate_normals()
point_cloud.orient_normals_to_align_with_direction()

print('run Poisson surface reconstruction')
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
o3d.visualization.draw_geometries([mesh])

o3d.io.write_triangle_mesh('../RESULTS/mesh_ortho.obj', mesh, write_triangle_uvs = True)
