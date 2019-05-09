import numpy as np 
import skimage 
from skimage import draw, measure 
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from skimage import img_as_float, io, measure
from skimage.draw import ellipse
from skimage.feature import corner_harris, corner_peaks, corner_subpix
from skimage.morphology import (closing, convex_hull_image, dilation, disk,
                                erosion, opening, white_tophat)
from skimage.transform import AffineTransform, warp
from skimage.util import invert
# --- Generate test data --- 
im = Image.open('test1.jpg')
im1 = im.filter(ImageFilter.GaussianBlur)

im2 = im.convert('L')  # 转化为灰图

im3 = ImageEnhance.Contrast(im2).enhance(100)


eselem = disk(10)

img = closing(im3, eselem)
# --- End generate test data --- 

# Generate a mask with all objects close to green 

mask = convex_hull_image(img)

# Label the different objects in the image 
label_img = measure.label(mask) 

# Find all objects 
regions = measure.regionprops(label_img) 

# Find the first object (since we only have one, that's the rectangle!) 
r = regions[0]

# We'll use the `bbox` property to select the bounding box 
# # For a full list of properties, take a look at the docstring of 
# measure.regionprops 

print('The bounding box is {}'.format(r.bbox)) 

r0, c0, r1, c1 = r.bbox 
img_crop = img[r0:r1, c0:c1] 

f, (ax0, ax1) = plt.subplots(1, 2) 
ax0.imshow(img) 
ax1.imshow(img_crop) 
plt.show() 