import cv2
import numpy as np
import matplotlib.pyplot as plt
from plane_sweep import plane_sweep_gauss, plane_sweep_ncc
from pathlib import Path
from PIL import Image

PATH_IMAGES = Path(__file__).resolve().parent.parent / 'data'

# Read images
left_path = PATH_IMAGES / "left.ppm"
right_path = PATH_IMAGES / "right.ppm"
IL = cv2.imread(str(left_path))   # left image
IR = cv2.imread(str(right_path))  # right image
im_l = np.array(Image.open(str(left_path)).convert('L'),'f')
im_r = np.array(Image.open(str(right_path)).convert('L'),'f')

print(IL.shape)

# intrinsic parameter matrix
fm = 403.657593 # Focal distantce in pixels
cx = 161.644318 # Principal point - x-coordinate (pixels) 
cy = 124.202080 # Principal point - y-coordinate (pixels) 
bl = 119.929 # baseline (mm)
# for the right camera    
right_k = np.array([[ fm, 0, cx],[0, fm, cy],[0, 0, 1.0000]])

# for the left camera
left_k = np.array([[fm, 0, cx],[0, fm, cy],[0, 0, 1.0000]])

# Extrinsic parameters
# Translation between cameras
T = np.array([-bl, 0, 0]) 
# Rotation
R = np.array([[ 1,0,0],[ 0,1,0],[0,0,1]])

print('Intrinsic Paramenters')
print('Left_K:\n', left_k)
print('Right_K:\n', right_k)

print('Extrinsic Paramenters')
print('R:\n', R)
print('T:\n', T)

steps = 45
start = 12

m,n = im_l.shape
print(m,n)

# width for ncc
wid1 = 9
wid2 = 3
res1 = plane_sweep_ncc(im_l,im_r,start,steps,wid1)
res2 = plane_sweep_gauss(im_l,im_r,start,steps,wid2)

plt.figure()
plt.imshow(res1,'gray')
plt.figure()
plt.imshow(res2,'gray')

Z = np.zeros((m,n))
for i in range(m):
	for j in range(n):
		if (res2[i,j]== 0):
			# Consider Z = inf for points that were not defined in the depthmap and are filled with zero
			Z[i,j] = np.inf
		else: Z[i,j] = fm * bl / res2[i,j]

# Prepare points to be 3D plotted
X,Y = np.meshgrid(np.arange(n),np.arange(m))
X = np.reshape(X, m*n)
Y = np.reshape(Y, m*n)
Z = np.reshape(Z, m*n)

x3d = ((X - cx) / fm) * Z
y3d = ((Y - cy) / fm) * Z

# Filter erroneous depths
good = np.where((Z>1000) & (Z<10000))

x3d = x3d[good]
y3d = y3d[good]
z3d = Z[good]

# Plotting Estimated and True 3D points
pixel_color = []

for i in range(X[good].shape[0]):
    pixel_color.append(IL[int(Y[good][i]), int(X[good][i])])
pixel_color = np.asarray(pixel_color)

# Show images 
fig, ax = plt.subplots(nrows=1, ncols=2)
plt.subplot(1, 2, 1)
plt.imshow(IL,cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(IR,cmap='gray')
plt.show(block=False)

# Plot 3D
fig = plt.figure(figsize=(10, 10))
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(x3d, y3d, z3d, c=pixel_color / 255.0)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.view_init(elev=-23, azim=-91)

# Plot 3D
fig = plt.figure(figsize=(10, 10))
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(x3d, y3d, z3d, c=pixel_color / 255.0)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.view_init(elev=-57, azim=-91)

plt.show()