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
