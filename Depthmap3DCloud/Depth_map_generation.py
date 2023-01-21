####################
##### This code is used for depth map and point cloud generation
#######################
import cv2
import torch
import matplotlib.pyplot as plt
import os
from os import listdir
from PIL import Image
import open3d as o3d
import random
import numpy as np

###### Here we use the model: https://github.com/isl-org/MiDaS:
###### Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

### check folder.... Example Scene 1
folder_dir=os.getcwd()+'/data/train/Scene1/'
folder_depth=os.getcwd()+'/data/train/depth/'

###### Or Scene 2
#folder_dir=os.getcwd()+'/data/test/Scene2/'
#folder_depth=os.getcwd()+'/data/train/depth/'

#### list images in the directory
image_directory=listdir(folder_dir)

##### get a random image and then
random.shuffle(image_directory)
print('Image ID:')
print(image_directory[0])
img = Image.open(folder_dir+image_directory[0])
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
inputs = feature_extractor(images=img, return_tensors="pt")
with torch.no_grad():
   outputs = model(**inputs)
predicted_depth = outputs.predicted_depth
# interpolate to original size
prediction = torch.nn.functional.interpolate(
predicted_depth.unsqueeze(1),
   size=img.size[::-1],
     mode="bicubic",
      align_corners=False,
              )    # visualize the prediction
output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[1].imshow(output, cmap='plasma')
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()

####### O3D LIbrary for point cloud generation.............
image = Image.open(folder_dir+image_directory[0])
width, height = image.size
depth_image = (output * 255 / np.max(output)).astype('uint8')
image = np.array(image)
# create rgbd image
depth_o3d = o3d.geometry.Image(depth_image)
image_o3d = o3d.geometry.Image(image)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

# camera settings
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)

# create point cloud
pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

### save Depth imges into folder
cv2.imwrite(folder_depth+image_directory[0],depth_image)


######### This code is taken from here for point cloud filtering and dispaly
##########
# outliers removal
cl, ind = pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
pcd = pointcloud.select_by_index(ind)

# estimate normals
pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction()

# surface reconstruction
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]

# rotate the mesh
rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
mesh.rotate(rotation, center=(0, 0, 0))

# save the mesh and one can see it meshlab
#o3d.io.write_triangle_mesh(f'folder_depth+image_directory[0]+'/mesh.obj', mesh)

# visualize the mesh
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)