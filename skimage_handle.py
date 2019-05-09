import matplotlib.pyplot as plt
import numpy as np
import skimage
from PIL import Image, ImageEnhance, ImageFilter
from skimage import draw, img_as_float, io, measure
from skimage import transform as tf
from skimage.draw import ellipse
from skimage.feature import corner_harris, corner_peaks, corner_subpix
from skimage.measure import (approximate_polygon, find_contours,
                             subdivide_polygon)
from skimage.morphology import (closing, convex_hull_image, dilation, disk,
                                erosion, opening, white_tophat)
from skimage.transform import AffineTransform, warp
from skimage.util import invert

TARGET_H = 697
TARGET_W = 1333


def get_max_contour_coor(img):
    max_contour = 0
    max_coor = []
    for contour in find_contours(img, 0):
        coord = approximate_polygon(contour, tolerance=39.5)
        print("Number of coordinates:", len(contour), len(coord))
        if len(contour) > max_contour:
            max_contour = contour
            max_coor = coord
        # 可以从上面的点个数判断票据位置状态
        print(coord)
        return len(max_coor), max_coor

# 四边形顶点排序，[top-left, top-right, bottom-right, bottom-left]


def orderPoints(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return np.flip(rect, 1)


# --- Generate test data ---
im = Image.open('test5.jpg')
reslotion_x, reslotion_y = im._size[0], im.size[1]
im1 = im.filter(ImageFilter.GaussianBlur)

im2 = im.convert('L')  # 转化为灰图

im3 = ImageEnhance.Contrast(im2).enhance(100)

eselem = disk(10)

img = closing(im3, eselem)
# --- End generate test data ---
point, coor = get_max_contour_coor(img)

if point <= 3:
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
    img_crop = im.crop((c0, r0, c1, r1))
    img_crop = img_crop.resize((TARGET_H, TARGET_W))

    f, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(im)
    ax1.imshow(img_crop)
    plt.show()
else:
    boxes = orderPoints(coor)
    target = np.array(
        [[0, 0], [0, TARGET_H], [TARGET_W, TARGET_H], [TARGET_W, 0]])
    tform3 = tf.ProjectiveTransform()
    tform3.estimate(target, boxes)
    warped = tf.warp(im, tform3, output_shape=(TARGET_H, TARGET_W))

    fig, ax = plt.subplots(nrows=2, figsize=(8, 3))

    ax[0].imshow(im, cmap=plt.cm.gray)
    ax[0].plot(boxes[:, 0], boxes[:, 1], '.r')
    ax[1].imshow(warped, cmap=plt.cm.gray)

    for a in ax:
        a.axis('off')

    plt.tight_layout()

    plt.show()
