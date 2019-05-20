#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import cv2

# 目标图片大小
TARGET_H = 672
TARGET_W = 1369
# 预加黑边大小
EDGE = 20


def get_binary_img(img, kernel_n):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化为灰图
    _, dst = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)  # otsu算法二值化
    kernel = np.ones((kernel_n, kernel_n), np.uint8)
    closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)  # 形态闭运算，去除票据内容
    # 下面加上四边黑框
    closed[:EDGE, :] = 0
    closed[-EDGE:, :] = 0
    closed[:, :EDGE] = 0
    closed[:, -EDGE:] = 0
    return closed


# 求出面积最大的轮廓
def findMaxContour(image):
    # 寻找边缘
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 计算面积
    max_area = 0.0
    max_contour = []
    for contour in contours:
        currentArea = cv2.contourArea(contour)
        if currentArea > max_area:
            max_area = currentArea
            max_contour = contour
    return max_contour, max_area


# 多边形拟合凸包的四个顶点
def getBoxPoint(contour):
    # 多边形拟合凸包
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    approx = approx.reshape((len(approx), 2))
    return approx


# 四边形顶点排序，[top-left, top-right, bottom-right, bottom-left]
def orderPoints(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 透视变换
def warpImage(image, box):
    dst_rect = np.array([[0, 0],
                         [TARGET_W - 1, 0],
                         [TARGET_W - 1, TARGET_H - 1],
                         [0, TARGET_H - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(box, dst_rect)
    warped = cv2.warpPerspective(image, M, (TARGET_W, TARGET_H))
    return warped


def main_img_preprocess(origin_img):
    shape = origin_img.shape
    binary_img = get_binary_img(origin_img, 10)
    max_contour, _ = findMaxContour(binary_img)
    boxes = getBoxPoint(max_contour)
    boxes = orderPoints(boxes)
    for co in boxes:
        if co[0] < EDGE+5:
            co[0] = 0
        if co[0] > shape[1] - EDGE - 5:
            co[0] = shape[1] - 1
        if co[1] < EDGE+5:
            co[1] = 0
        if co[1] > shape[0] - EDGE - 5:
            co[1] = shape[0] - 1

    warped = warpImage(img, boxes)
    return warped


# path = '4.jpg'
# img = cv2.imread(path)
# crop_img = main_img_preprocess(img)
# cv2.imwrite('result.jpg', crop_img)
# cv2.imshow('warpImage', crop_img)
# cv2.waitKey(0)
