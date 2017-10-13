import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import misc
from scipy.ndimage import filters


# low pass filter
img1 = misc.imread('einstein.bmp',flatten=1)

img1_blur = ndimage.gaussian_filter(img1,21)
img1_sharpen = img1 - img1_blur
img1_blur = ndimage.gaussian_filter(img1_sharpen,4)

plt.imshow(img1_blur,cmap=plt.cm.gray)
plt.show()

# high pass filter
img2 = misc.imread('marilyn.bmp',flatten=1)
# kernel = np.array([[-2, -2, -2],
#                    [-2,  16, -2],
#                    [-2, -2, -2]])
# #
# # kernel = np.array([[-1, -1, -1, -1, -1],
# #                    [-1,  1,  2,  1, -1],
# #                    [-1,  2,  4,  2, -1],
# #                    [-1,  1,  2,  1, -1],
# #                    [-1, -1, -1, -1, -1]])
#
# img2_sharpen = filters.convolve(img2,kernel,mode='constant')


img2_blur = ndimage.gaussian_filter(img2,7)
img2_sharpen = img2 - img2_blur
img2_sharpen = ndimage.gaussian_filter(img2_sharpen,2)
plt.imshow(img2_sharpen,cmap=plt.cm.gray)
plt.show()

img3 = img1_blur+img2_sharpen
plt.imshow(img3,cmap=plt.cm.gray)
plt.show()

plt.subplot(1, 3, 1)
plt.title('image1_blurred', fontsize=10)
plt.imshow(img1_blur, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1, 3, 2)
plt.title('image2_sharpend', fontsize=10)
plt.imshow(img2_sharpen, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1, 3, 3)
plt.title('image3', fontsize=10)
plt.imshow(img3, cmap=plt.cm.gray)
plt.axis('off')

plt.show()


# pyramids

img = misc.imresize(img3,(256,256))


kernel = (1.0/256)*np.array([[1, 4,  6,  4,  1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6],[4, 16, 24, 16, 4],[1, 4,  6,  4,  1]])


#img_up = np.zeros((2*img.shape[0], 2*img.shape[1]))
#img_up[::2, ::2] = img
#ndimage.filters.convolve(img_up,4*kernel, mode='constant')

#sig.convolve2d(img_up, 4*kernel, 'same')

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

#interpolate(img)
#decimate(img)
[G,L] = pyramids(img)

rows, cols = img.shape
composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
composite_image[:rows, :cols] = G[0]

i_row = 0
for p in G[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows


fig, ax = plt.subplots()

ax.imshow(composite_image,cmap='gray')
plt.show()


# rows, cols = img.shape
# composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
#
# composite_image[:rows, :cols] = L[0]
#
# i_row = 0
# for p in L[1:]:
#     n_rows, n_cols = p.shape[:2]
#     composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
#     i_row += n_rows
#
#
# fig, ax = plt.subplots()
#
# ax.imshow(composite_image,cmap='gray')
# plt.show()
