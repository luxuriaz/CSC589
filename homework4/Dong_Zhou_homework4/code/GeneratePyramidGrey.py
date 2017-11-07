# -*- coding: utf-8 -*-
"""

@author: Dong Zhou

"""
import numpy as np
import scipy.signal as sig
from scipy import misc
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2


def interpolate(image):
    """
    Interpolates an image with upsampling rate r=2.
    """
    image_up = np.zeros((2*image.shape[0], 2*image.shape[1]))
    # Upsample
    image_up[::2, ::2] = image
    # Blur (we need to scale this up since the kernel has unit area)
    # (The length and width are both doubled, so the area is quadrupled)
    #return sig.convolve2d(image_up, 4*kernel, 'same')
    return ndimage.filters.convolve(image_up,4*kernel, mode='constant')

def decimate(image):
    """
    Decimates at image with downsampling rate r=2.
    """
    # Blur
    #image_blur = sig.convolve2d(image, kernel, 'same')
    image_blur = ndimage.filters.convolve(image,kernel, mode='constant')
    # Downsample
    return image_blur[::2, ::2]


# here is the constructions of pyramids
def pyramids(image):
    """
    Constructs Gaussian and Laplacian pyramids.
    Parameters :
        image  : the original image (i.e. base of the pyramid)
    Returns :
        G   : the Gaussian pyramid
        L   : the Laplacian pyramid
    """
    # Initialize pyramids
    G = [image, ]
    L = []

    # Build the Gaussian pyramid to maximum depth
    while image.shape[0] >= 2 and image.shape[1] >= 2:
        image = decimate(image)
        G.append(image)

    # Build the Laplacian pyramid
    for i in range(len(G) - 1):
        L.append(G[i] - interpolate(G[i + 1]))

    return G[:-1], L


'''
Blend the two laplacian pyramids by weighting them according to the mask.
From https://compvisionlab.wordpress.com/2013/05/13/image-blending-using-pyramid/

'''
def blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
  blended_pyr = []
  k= len(gauss_pyr_mask)
  for i in range(0,k):
   p1= gauss_pyr_mask[i]*lapl_pyr_white[i]
   p2=(1 - gauss_pyr_mask[i])*lapl_pyr_black[i]
   p3 = p1+p2
   # plt.imshow(p3,cmap='gray')
   # plt.show()
   blended_pyr.append(p3)
  return blended_pyr
'''
Reconstruct the image based on its laplacian pyramid.
From https://compvisionlab.wordpress.com/2013/05/13/image-blending-using-pyramid/
modified by Dong Zhou
'''
def collapse(lapl_pyr):
  output = None
  output = np.zeros((lapl_pyr[0].shape[0],lapl_pyr[0].shape[1]), dtype=np.float64)
  for i in range(len(lapl_pyr)-1,0,-1):
    lap = interpolate(lapl_pyr[i])
    lapb = lapl_pyr[i-1]
    if lap.shape[0] > lapb.shape[0]:
      lap = np.delete(lap,(-1),axis=0)
    if lap.shape[1] > lapb.shape[1]:
      lap = np.delete(lap,(-1),axis=1)
    tmp = lap + lapb
    lapl_pyr.pop()
    lapl_pyr.pop()
    lapl_pyr.append(tmp)
    output = tmp
  return output

def split_rgb(image):
  blue, green, red    = image[:, :, 0], image[:, :, 1], image[:, :, 2]
  # (blue,green,red)=cv2.split(image)
  return (red,green,blue)

def combine(r,g,b):
    result = np.zeros(img.shape,dtype=img.dtype)
    colors = []
    colors.append(b)
    colors.append(g)
    colors.append(r)
    result = np.dstack(colors)
    return result

# read image
img = misc.imread('apple.jpg',flatten=1)
img1 = misc.imread('orange.jpg',flatten=1)
mask = misc.imread('mask.jpg',flatten=1)
img = img.astype(float)
img1 = img1.astype(float)
mask = mask.astype(float)/255

# create a  Binomial (5-tap) filter
kernel = (1.0/256)*np.array([[1, 4,  6,  4,  1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6],[4, 16, 24, 16, 4],[1, 4,  6,  4,  1]])

plt.imshow(kernel)
plt.show()
'''split RBG'''

# (r1,g1,b1) = split_rgb(img)
# (r2,g2,b2) = split_rgb(img1)
# (rm,gm,bm) = split_rgb(mask)
#
# r1 = r1.astype(float)
# g1 = g1.astype(float)
# b1 = b1.astype(float)
#
# r2 = r2.astype(float)
# g2 = g2.astype(float)
# b2 = b2.astype(float)
#
# rm = rm.astype(float)/255
# gm = gm.astype(float)/255
# bm = bm.astype(float)/255

'''Gassian and Lapacian of apple image'''
[G_apple,L_apple] = pyramids(img)

# show the laplacian pyramids of apple
rows, cols = img.shape
composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
composite_image[:rows, :cols] = L_apple[0]

i_row = 0
for p in L_apple[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows


fig, ax = plt.subplots()

ax.imshow(composite_image,cmap='gray')
plt.show()

'''Gassian and Lapacian of orange image'''

[G_orange,L_orange] = pyramids(img1)


# show the laplacian pyramids of orange
rows, cols = img.shape
composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
composite_image[:rows, :cols] = L_orange[0]

i_row = 0
for p in L_orange[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows


fig, ax = plt.subplots()

ax.imshow(composite_image,cmap='gray')
plt.show()

'''Gassian and Lapacian of mask image'''

[G_mask,L_mask] = pyramids(mask)
# show the laplacian pyramids of mask

rows, cols = img.shape
composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
composite_image[:rows, :cols] = G_mask[0]

i_row = 0
for p in G_mask[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows


fig, ax = plt.subplots()

ax.imshow(composite_image,cmap='gray')
plt.show()

# reconstruct the pyramids, here you write a reconstrut function that takes the
# pyramid and upsampling the each level and add them up.


blend_pyramid = blend(L_orange,L_apple,G_mask)


rows, cols = img.shape
composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
composite_image[:rows, :cols] = blend_pyramid[0]

i_row = 0
for p in blend_pyramid[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows


fig, ax = plt.subplots()

ax.imshow(composite_image,cmap='gray')
plt.show()

# blend_image = collapse(blend(L_orange,L_apple,G_mask))


blend_image = collapse(blend_pyramid)
plt.imshow(blend_image,cmap=plt.cm.gray)
plt.show()
cv2.imwrite('blended.jpg', blend_image)
