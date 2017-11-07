import numpy as np
import scipy.signal as sig
from scipy import misc
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2



'''
Problem 1

'''


img = misc.imread('steam.jpg',flatten=1)
img = img.astype(float)
print img.shape



# kernel = np.array([[1, 4,  6,  4,  1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6],[4, 16, 24, 16, 4],[1, 4,  6,  4,  1]])
ncols, nrows = img.shape
sigmax, sigmay = 20, 20
cy, cx = nrows/2, ncols/2
x = np.linspace(0, nrows, nrows)
y = np.linspace(0, ncols, ncols)
X, Y = np.meshgrid(x, y)
kernel = np.exp(-(((X-cx)/sigmax)**2 + ((Y-cy)/sigmay)**2))
plt.imshow(kernel,cmap=plt.cm.gray)
plt.show()

img_blur = sig.fftconvolve(img,kernel,mode="full")

f_img = np.fft.fft2(img)

# fshift_img_blur = np.fft.fftshift(f_img_blur)

f_kernel = np.fft.fft2(kernel)
f_img_blur = f_img * f_kernel

# fshift_kernel = np.fft.fftshift(f_kernel)

# # f_origin  = f_img_blur / f_kernel
# # original = np.fft.ifft2(f_origin)

o_shift = f_img_blur / f_kernel

o_shift = np.fft.fftshift(o_shift)

original = np.fft.ifft2(o_shift)

original = original.astype(float)

# print original.shape

plt.figure(figsize=(15, 4))
plt.subplot(131)
plt.imshow(img, cmap=plt.cm.gray)
plt.subplot(132)
plt.imshow(original,cmap=plt.cm.gray)
plt.subplot(133)
plt.imshow(img_blur, cmap=plt.cm.gray)
plt.show()

'''
The deconvolved image has many noise compare to the original image,
I think it is because we ignored the phase shift when we do the
inverse Fourier transform.
'''



'''
Problem 2

'''
def rgb(image):
    blue, green, red    = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    # (blue,green,red)=cv2.split(image)
    return red,green,blue


def median_filter(img):

    # create same size empty image
    median_img = np.matlib.repmat(0,img.shape[0],img.shape[1])
    # do the median filtering
    filtering=[img[0,0]]*9
    for y in range(1,img.shape[0]-1):
        for x in range(1,img.shape[1]-1):
            filtering[0] = img[y-1,x-1]
            filtering[1] = img[y,x-1]
            filtering[2] = img[y+1,x-1]
            filtering[3] = img[y-1,x]
            filtering[4] = img[y,x]
            filtering[5] = img[y+1,x]
            filtering[6] = img[y-1,x+1]
            filtering[7] = img[y,x+1]
            filtering[8] = img[y+1,x+1]
            filtering.sort()
            median_img[y,x]=filtering[4]

    # plt.figure(figsize=(8, 4))
    # plt.subplot(121)
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.subplot(122)
    # plt.imshow(median_img,cmap=plt.cm.gray)
    # plt.show()
    return median_img

def combine(r,g,b):
    result = np.zeros(img.shape,dtype=img.dtype)
    colors = []
    colors.append(b)
    colors.append(g)
    colors.append(r)
    result = np.dstack(colors)
    return result

img = misc.imread('taj-rgb-noise.jpg')
(r,g,b) = rgb(img)
r = r.astype(float)
g = g.astype(float)
b = b.astype(float)
print r.shape
r_median = median_filter(r)
g_median = median_filter(g)
b_median = median_filter(b)
r_median = r_median.astype(np.uint8)
g_median = g_median.astype(np.uint8)
b_median = b_median.astype(np.uint8)
result = combine(r_median,g_median,b_median)
result = np.asarray(result)
# print result
# print result.shape
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.imshow(img, cmap=plt.cm.gray)
plt.subplot(122)
plt.imshow(result,cmap=plt.cm.gray)
plt.show()
