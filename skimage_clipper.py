import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure, filters, io, measure, transform
from skimage.color import rgb2gray
from skimage.morphology import closing, convex_hull_image, disk

TARGET_H = 672
TARGET_W = 1369


def get_binary_img(img, kernel):
    img = rgb2gray(img)  # 转化为灰图
    io.imshow(img)
    plt.show()
    grayscale = exposure.adjust_gamma(img, 3)
    selem = disk(kernel)
    closed = closing(grayscale, selem)
    thresh = filters.threshold_otsu(closed)
    dst = (closed >= thresh)*1.0
    io.imshow(dst)
    plt.show()
    return dst


def get_max_contour_coor(img):
    max_contour = 0
    max_coor = []
    for contour in measure.find_contours(img, 0):
        coord = measure.approximate_polygon(contour, tolerance=39.5)
        if len(contour) > max_contour:
            max_contour = len(contour)
            max_coor = coord
        # 可以从上面的点个数判断票据位置状态
    return len(max_coor), max_coor


# 四边形顶点排序，[top-left,bottom-left, top-right, bottom-right ]
def orderPoints(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return np.flip(rect, 1)


def get_crop_img(origin_img, binary_img, point, coor):
    img = binary_img
    if point <= 3:
        # 如果轮廓点不大于3点，则票据必有至少一边和图片边重合，此时不需要透视拉伸转换，直接裁剪即可。
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

        r0, c0, r1, c1 = r.bbox
        img_crop = origin_img[r0:r1, c0:c1]
        resized = transform.resize(img_crop, (TARGET_H, TARGET_W))
        # io.imshow(resized)
        # plt.show()
        return resized
    else:
        # 如果轮廓点为4点以上，则找轮廓点并进行透视转换
        boxes = orderPoints(coor)
        target = np.array(
            [[0, 0], [0, TARGET_H], [TARGET_W, TARGET_H], [TARGET_W, 0]])
        tform3 = transform.ProjectiveTransform()
        tform3.estimate(target, boxes)
        warped = transform.warp(
            origin_img, tform3, output_shape=(TARGET_H, TARGET_W))
        return warped


origin_img = io.imread('test.jpg')
img = get_binary_img(origin_img, 10)
point, coor = get_max_contour_coor(img)
crop_img = get_crop_img(origin_img, img, point, coor)
# io.imshow(crop_img)
# plt.show()
io.imsave('result.jpg', crop_img)
