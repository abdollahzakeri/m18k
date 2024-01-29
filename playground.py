import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

matplotlib.use("TkAgg")
image = cv2.imread('M18K/Data/train/880_1680293712-6002584_rgb_png_jpg.rf.41ce28d3bfbe0d04a5197f874eba9d78.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
depth_map = np.load('M18K/Data/train/880_1680293712-6002584_rgb_png_jpg.rf.41ce28d3bfbe0d04a5197f874eba9d78.npy')
percentiles = np.percentile(depth_map, range(1, 101))
differences = np.diff(percentiles)
max_diff_index = np.argmax(differences)
threshold = percentiles[max_diff_index]
if max_diff_index > 95 and threshold > 0:
    depth_map = np.clip(depth_map, 0.0, threshold)
depth_map[depth_map == 0] = np.mean(depth_map)
depth_map = median_filter(depth_map, size=15)
depth_map = np.max(depth_map) - depth_map
assert image.shape[:2] == depth_map.shape, "Image and depth map sizes do not match"
height, width = image.shape[:2]
x, y = np.meshgrid(np.arange(width), np.arange(height))
z = depth_map
valid = depth_map > 0
x, y, z = x[valid], y[valid], z[valid]
colors = image[valid]
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=colors.reshape(-1, 3) / 255, marker='.')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
