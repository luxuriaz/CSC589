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
   blended_pyr.append(p1+p2)
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
  (blue, green, red) = cv2.split(image)
  return red, green, blue

def combine(r,g,b):
    result = np.zeros(img.shape,dtype=img.dtype)
    colors = []
    colors.append(b)
    colors.append(g)
    colors.append(r)
    result = cv2.merge(colors,result)
    return result

# read image
img = cv2.imread('apple.jpg')
img1 = cv2.imread('orange.jpg')
mask = cv2.imread('mask.jpg')
img = img.astype(float)
img1 = img1.astype(float)
mask = mask.astype(float)/255

# create a  Binomial (5-tap) filter
def generating_kernel(a):
  w_1d = np.array([0.25 - a/2.0, 0.25, a, 0.25, 0.25 - a/2.0])
  return np.outer(w_1d, w_1d)

kernel = generating_kernel(0.4)

# kernel = (1.0/256)*np.array([[1, 4,  6,  4,  1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6],[4, 16, 24, 16, 4],[1, 4,  6,  4,  1]])



# plt.imshow(kernel)
# plt.show()
'''split RBG'''

(r1,g1,b1) = split_rgb(img)
(r2,g2,b2) = split_rgb(img1)
(rm,gm,bm) = split_rgb(mask)

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
[G_apple_r,L_apple_r] = pyramids(r1)
[G_apple_g,L_apple_g] = pyramids(g1)
[G_apple_b,L_apple_b] = pyramids(b1)
# show the laplacian pyramids of apple
# rows, cols = 512,512
# composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
# composite_image[:rows, :cols] = L_apple_r[0]
#
# i_row = 0
# for p in L_apple_r[1:]:
#     n_rows, n_cols = p.shape[:2]
#     composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
#     i_row += n_rows
#
#
# fig, ax = plt.subplots()
#
# ax.imshow(composite_image,cmap='gray')
# plt.show()

'''Gassian and Lapacian of orange image'''

[G_orange_r,L_orange_r] = pyramids(r2)
[G_orange_g,L_orange_g] = pyramids(g2)
[G_orange_b,L_orange_b] = pyramids(b2)

# show the laplacian pyramids of orange
# rows, cols = 512,512
# composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
# composite_image[:rows, :cols] = L_orange_r[0]
#
# i_row = 0
# for p in L_orange_r[1:]:
#     n_rows, n_cols = p.shape[:2]
#     composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
#     i_row += n_rows
#
#
# fig, ax = plt.subplots()
#
# ax.imshow(composite_image,cmap='gray')
# plt.show()

'''Gassian and Lapacian of mask image'''

[G_mask_r,L_mask_r] = pyramids(rm)
[G_mask_g,L_mask_g] = pyramids(gm)
[G_mask_b,L_mask_b] = pyramids(bm)

# show the laplacian pyramids of mask

# rows, cols = 512,512
# composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
# composite_image[:rows, :cols] = G_mask_r[0]
#
# i_row = 0
# for p in G_mask_r[1:]:
#     n_rows, n_cols = p.shape[:2]
#     composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
#     i_row += n_rows
#
#
# fig, ax = plt.subplots()
#
# ax.imshow(composite_image,cmap='gray')
# plt.show()

# reconstruct the pyramids, here you write a reconstrut function that takes the
# pyramid and upsampling the each level and add them up.


blend_pyramid_r = blend(L_orange_r,L_apple_r,G_mask_r)
blend_pyramid_g = blend(L_orange_g,L_apple_g,G_mask_g)
blend_pyramid_b = blend(L_orange_b,L_apple_b,G_mask_b)
#
#
# rows, cols = 512,512
# composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
# composite_image[:rows, :cols] = blend_pyramid_r[0]
#
# i_row = 0
# for p in blend_pyramid_r[1:]:
#     n_rows, n_cols = p.shape[:2]
#     composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
#     i_row += n_rows
#
#
# fig, ax = plt.subplots()
#
# ax.imshow(composite_image,cmap='gray')
# plt.show()
#
# # blend_image = collapse(blend(L_orange,L_apple,G_mask))
#
#
blend_image_r = collapse(blend_pyramid_r)
blend_image_g = collapse(blend_pyramid_g)
blend_image_b = collapse(blend_pyramid_b)

blend_image_r = blend_image_r.astype(np.uint8)
blend_image_g = blend_image_g.astype(np.uint8)
blend_image_b = blend_image_b.astype(np.uint8)
#
#
blend_image = combine(blend_image_r,blend_image_g,blend_image_b)
#
# plt.imshow(blend_image,cmap=plt.cm.gray)
# plt.show()
cv2.imwrite('blended.jpg', blend_image)
