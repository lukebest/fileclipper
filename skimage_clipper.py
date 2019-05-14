# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure, filters, io, measure, transform
from skimage.color import rgb2gray
from skimage.morphology import closing, convex_hull_image, disk


# 目标图片大小
TARGET_H = 672
TARGET_W = 1369
# 预加黑边大小
EDGE = 20


def get_binary_img(img, kernel):
    img = rgb2gray(img)  # 转化为灰图
    thresh = filters.threshold_otsu(img)  # otsu算法二值化
    dst = (img >= thresh)*1.0
    selem = disk(kernel)
    closed = closing(dst, selem)  # 形态闭运算，去除票据内容
    # 下面加上四边黑框
    closed[:EDGE, :] = 0.
    closed[-EDGE:, :] = 0.
    closed[:, :EDGE] = 0.
    closed[:, -EDGE:] = 0.
    # io.imshow(closed)
    # plt.show()
    return closed


def get_max_contour_coor(img):
    max_contour = 0
    max_coor = []
    # 寻找最大轮廓
    for contour in measure.find_contours(img, 0):
        # tolerance75 寻找矩形轮廓，忽略可能存在的反光斑突起
        coord = measure.approximate_polygon(contour, tolerance=75)
        if len(contour) > max_contour:
            max_contour = len(contour)
            max_coor = coord
    return len(max_coor), max_coor


# 轮廓四边形顶点排序输出用于透视变换
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
    shape = origin_img.shape
    # 如果轮廓在黑边上，则减掉黑边回到原图片对应角点。
    for co in coor:
        if co[0] < EDGE+5:
            co[0] = 0
        if co[0] > shape[0] - EDGE - 5:
            co[0] = shape[0] - 1
        if co[1] < EDGE+5:
            co[1] = 0
        if co[1] > shape[1] - EDGE - 5:
            co[1] = shape[1] - 1

    boxes = orderPoints(coor)
    target = np.array(
        [[0, 0], [0, TARGET_H], [TARGET_W, TARGET_H], [TARGET_W, 0]])
    # 进行透视变换
    tform3 = transform.ProjectiveTransform()
    tform3.estimate(target, boxes)
    warped = transform.warp(
        origin_img, tform3, output_shape=(TARGET_H, TARGET_W))
    return warped


def main_img_preprocess(origin_img):
    bin_img = get_binary_img(origin_img, 10)
    point, coor = get_max_contour_coor(bin_img)
    crop_img = get_crop_img(origin_img, bin_img, point, coor)
    return crop_img


# origin_img = io.imread('test5.jpg')
# crop_img = main_img_preprocess(origin_img)
# io.imshow(crop_img)
# plt.show()
# io.imsave('result.jpg', crop_img)
