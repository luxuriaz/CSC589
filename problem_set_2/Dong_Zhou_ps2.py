from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#
# Problem 1 warm up
einstein_i = misc.imread('einstein.png',flatten=1)
einstein_G = ndimage.gaussian_filter(einstein_i,3)
plt.imshow(einstein_G,cmap=plt.cm.gray)
plt.show()

zebra_i = misc.imread('zebra.png',flatten=1)
zebra_G = ndimage.gaussian_filter(zebra_i,3)
plt.imshow(zebra_G,cmap=plt.cm.gray)
plt.show()

# DFT

f = np.fft.fft2(einstein_G)
fshift = np.fft.fftshift(f)
magnitude = 20*np.log(np.abs(fshift))


# f.shape()
# magnitudes of DFT
# m = abs(f)

plt.plot(magnitude)
plt.show()
# print m
#
# # Problem 2
#
# problem2_i = misc.imread('problem2.jpeg',flatten=1)
# plt.imshow(problem2_i,cmap=plt.cm.gray)
# plt.show()
#
# imhist,bin_edges = np.histogram(problem2_i.flatten(),normed=True)
# cdf = imhist.cumsum()
# plt.plot(cdf)
# plt.show()
# plt.plot(imhist)
# plt.show()
#
#
# #normalize
# normalized_cdf = 255 * cdf / cdf[-1]
# plt.plot(normalized_cdf)
# # plt.axis([0,10,0, 255])
# plt.show()
# # print normalized_cdf
#
# # Problem 3
# # DID NOT UNDERSTAND

# # Problem 4
#
# l = misc.imread('peppers.png',flatten=1)
#
# # image derivative
# # a) set up filters
# # derivative
# dx = np.array([1,-1])
# s1 = np.array([1,1])
# dy = np.array([1,-1])
# # smooth filter
# s =s1[None,:]
#
# # convolve with the image using above filters
# x = ndimage.convolve1d(l,dx,axis=0)
# gx_I=ndimage.convolve(x,s)
#
# # plt.imshow(gx_I,cmap=plt.cm.gray)
# # plt.show()
#
# y = ndimage.convolve1d(l,dx,axis=1)
# gy_I=ndimage.convolve(y,s)
#
# xy = (gy_I + gx_I)*(0.5)
# plt.imshow(xy,cmap=plt.cm.gray)
# plt.title('gardients filter', fontsize=20)
# plt.show()
#
#
#
# # sx = ndimage.sobel(x,axis= 0,mode = 'constant')
# # sy = ndimage.sobel(y,axis= 1,mode = 'constant')
# # sob_1 = np.hypot(sx, sy)
# # plt.imshow(gy_I,cmap=plt.cm.gray)
# # plt.show()
# # plt.imshow(sob,cmap=plt.cm.gray)
# # plt.show()
#
# sx = ndimage.sobel(gx_I,axis= 0,mode = 'constant')
# sy = ndimage.sobel(gy_I,axis= 1,mode = 'constant')
# sob = np.hypot(sx, sy)
#
# plt.imshow(sob,cmap=plt.cm.gray)
# plt.title('gardients/sob filter', fontsize=20)
# plt.show()
#
#
# plt.figure(figsize=(15, 4))
# plt.subplot(131)
# plt.imshow(l, cmap=plt.cm.gray)
# plt.subplot(132)
# plt.imshow(xy,cmap=plt.cm.gray)
# plt.subplot(133)
# plt.imshow(sob, cmap=plt.cm.gray)
# # plt.axis('off')
# plt.show()
