import cv2
from sympy import *
import scipy.integrate
from numpy import exp
from math import sqrt
import math
import numpy as np
from matplotlib import pyplot as plt
import pylab

def toTl(I):
    if I<=127:
        return 17 * (1 - sqrt(I / 127)) + 3
    else:
        return 3 / 128 * (I - 127) + 3

def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def agcwd(image, w=0.25):
    is_colorful = len(image.shape) >= 3
    img = extract_value_channel(image) if is_colorful else image
    img_pdf = get_pdf(img)
    max_intensity = np.max(img_pdf)
    min_intensity = np.min(img_pdf)
    w_img_pdf = max_intensity * (((img_pdf - min_intensity) / (max_intensity - min_intensity)) ** w)
    w_img_cdf = np.cumsum(w_img_pdf) / np.sum(w_img_pdf)
    l_intensity = np.arange(0, 256)
    l_intensity = np.array([255 * (e / 255) ** (1 - w_img_cdf[e]) for e in l_intensity], dtype=np.uint8)
    enhanced_image = np.copy(img)
    height, width = img.shape
    for i in range(0, height):
        for j in range(0, width):
            intensity = enhanced_image[i, j]
            enhanced_image[i, j] = l_intensity[intensity]
    enhanced_image = set_value_channel(image, enhanced_image) if is_colorful else enhanced_image
    return enhanced_image


def extract_value_channel(color_image):
    color_image = color_image.astype(np.float32) / 255.
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    return np.uint8(v * 255)


def get_pdf(gray_image):
    height, width = gray_image.shape
    pixel_count = height * width
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    return hist / pixel_count


def set_value_channel(color_image, value_channel):
    value_channel = value_channel.astype(np.float32) / 255
    color_image = color_image.astype(np.float32) / 255.
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    color_image[:, :, 2] = value_channel
    color_image = np.array(cv2.cvtColor(color_image, cv2.COLOR_HSV2BGR) * 255, dtype=np.uint8)
    return color_image


image = cv2.imread(r'C:\Users\11021\Desktop\a36.jpg')
imgShape = image.shape  # 大小/尺寸
x_range = imgShape[0]
y_range = imgShape[1]

HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(HSV)

V = replaceZeroes(V)
print(1)


sigma = [15, 80, 250]

I = np.empty([x_range, y_range, 3], dtype=np.float32)
for z in range(3):
    I[:, :, z] = cv2.GaussianBlur(V, (0, 0), sigma[z])
print(1)

T = np.empty([x_range, y_range, 3], dtype = np.float32)
beta = np.empty([x_range, y_range, 3], dtype = np.float32)
k = 0.3
for i in range(x_range):
    for j in range(y_range):
        for z in range(3):
            T[i, j, z] = toTl(I[i, j, z])
            beta[i, j, z] = k * (-(1 / 17 * T[i, j, z]) + 20 / 17)

print(1)


r = np.zeros((x_range, y_range), dtype=np.float32)
R = np.empty([x_range, y_range], dtype=np.float32)
Rmax = 0.0
Rmin = 1000000.0
w = [1/3, 1/3, 1/3]
for i in range(x_range):
    for j in range(y_range):
        for z in range(3):
            r[i, j] += w[z] * (math.log(V[i, j]) - beta[i, j, z] * math.log(I[i, j, z]))
        R[i, j] = exp(r[i, j])
        if R[i, j] > Rmax:
            Rmax = R[i, j]
        if R[i, j] < Rmin:
            Rmin = R[i, j]

print(2)


Vw = np.empty([x_range, y_range], dtype=np.float32)
Va = np.empty([x_range, y_range], dtype=np.uint8)
for i in range(x_range):
    for j in range(y_range):
        Vw[i, j] = (R[i, j] - Rmin) / (Rmax - Rmin)
        Va[i, j] = int(Vw[i, j]*255)


Vout = agcwd(Va)

Iout2 = cv2.merge([H, S, Vout])
IMG = cv2.cvtColor(Iout2, cv2.COLOR_HSV2BGR)
cv2.imshow("image", IMG)

cv2.waitKey(0)
cv2.destroyAllWindows()
