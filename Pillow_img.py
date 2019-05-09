import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from skimage import img_as_float, io, measure
from skimage.draw import ellipse
from skimage.feature import corner_harris, corner_peaks, corner_subpix
from skimage.morphology import (closing, convex_hull_image, dilation, disk,
                                erosion, opening, white_tophat)
from skimage.transform import AffineTransform, warp
from skimage.util import invert

im = Image.open('test.jpg')
im1 = im.filter(ImageFilter.GaussianBlur)

im2 = im.convert('L')  # 转化为灰图

im3 = ImageEnhance.Contrast(im2).enhance(100)


eselem = disk(10)

opened = closing(im3, eselem)


# The original image is inverted as the object must be white.
image = opened
coords = corner_peaks(corner_harris(image), min_distance=25)
coords_subpix = corner_subpix(image, coords, window_size=13)

fig, ax = plt.subplots()
ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)
ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
ax.axis()
plt.show()
