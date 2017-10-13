import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import misc
from scipy.ndimage import filters

img1 = misc.imread('dog.bmp', mode="RGB")
img2 = misc.imread('cat.bmp',mode="RGB")


#img1
I1 = img1.copy()  # Duplicate image


# blur each color channel

I_red = I1[:, :, 0]
I_red_blur = ndimage.gaussian_filter(I_red,7)

I_green = I1[:, :, 1]
I_green_blur = ndimage.gaussian_filter(I_green,7)

I_blue = I1[:, :, 2]
I_blue_blur = ndimage.gaussian_filter(I_blue,7)

# put each color channel back after blurring

I1[:, :, 0] = I_red_blur
I1[:, :, 1] = I_green_blur
I1[:, :, 2] = I_blue_blur


plt.subplot(1, 4, 1)
plt.title('red blur', fontsize=10)
plt.imshow(I_red_blur , cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1, 4, 2)
plt.title('green blur', fontsize=10)
plt.imshow(I_green_blur, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1, 4, 3)
plt.title('blue blur', fontsize=10)
plt.imshow(I_blue_blur, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1, 4, 4)
plt.title('low pass filter', fontsize=10)
plt.imshow(I1, cmap=plt.cm.gray)
plt.axis('off')
plt.show()


#img2
I2 = img2.copy()  # Duplicate image


# sharpen each color channel
I_red = I2[:, :, 0]
I_red_blur = ndimage.gaussian_filter(I_red,2)
I_red_sharpen = I_red - I_red_blur


I_green = I2[:, :, 1]
I_green_blur = ndimage.gaussian_filter(I_green,2)
I_green_sharpen = I_green - I_green_blur

I_blue = I2[:, :, 2]
I_blue_blur = ndimage.gaussian_filter(I_blue,2)
I_blue_sharpen = I_blue - I_blue_blur

# put each color channel back after sharpening

I2[:, :, 0] = I_red_sharpen
I2[:, :, 1] = I_green_sharpen
I2[:, :, 2] = I_blue_sharpen

plt.subplot(1, 4, 1)
plt.title('red sharpen', fontsize=10)
plt.imshow(I_red_sharpen , cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1, 4, 2)
plt.title('green sharpen', fontsize=10)
plt.imshow(I_green_sharpen, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1, 4, 3)
plt.title('blue sharpen', fontsize=10)
plt.imshow(I_blue_sharpen, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1, 4, 4)
plt.title('high pass filter', fontsize=10)
plt.imshow(I2, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
