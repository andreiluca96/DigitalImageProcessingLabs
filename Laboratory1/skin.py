import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


def plot_figures(figures, nrows=1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()  # optional


def is_rgb_pixel_skin(pixel, mod):
    b = int(pixel[0])
    g = int(pixel[1]) + 0.00000001
    r = int(pixel[2])

    if mod == 1:
        if r > 95 and g > 40 and b > 20 and \
                max(pixel) - min(pixel) > 15 and np.abs(r - g) > 15 and \
                r > g and r > b:
            return [255, 255, 255]
        return [0, 0, 0]
    if mod == 2:
        if r / g > 1.185 and \
                r * b / (r ** 2 + g ** 2 + b ** 2) > 0.107 and \
                r * g / (r ** 2 + g ** 2 + b ** 2) > 0.112:
            return [255, 255, 255]
        return [0, 0, 0]


def is_hsv_pixel_skin(pixel, mod):
    h = pixel[0]
    s = pixel[1] / 255
    v = pixel[2] / 180

    # print(f'{h} {s} {v}')

    if mod == 1:
        if v >= 0.4 and 0.2 < s < 0.6 and (0 < h < 25 or 335 < h <= 360):
            return [255, 255, 255]
        return [0, 0, 0]

    if mod == 2:
        if 0 <= h <= 50 and 0.23 <= s <= 0.68 and 0.35 <= v <= 1:
            return [255, 255, 255]
        return [0, 0, 0]

    if mod == 3:
        if (0 < h < 50 or 340 < h < 360) and s >= 0.2 and v >= 0.35:
            return [255, 255, 255]
        return [0, 0, 0]


def is_ycbcr_pixel_skin(pixel, mod):
    y = pixel[0]
    cb = pixel[1]
    cr = pixel[2]

    if mod == 1:
        if y > 80 and 85 < cb < 135 < cr < 180:
            return [255, 255, 255]
        return [0, 0, 0]

    if mod == 2:
        if 77 <= cb < 127 and 133 <= cr <= 173:
            return [255, 255, 255]
        return [0, 0, 0]


def draw_face_shape(img):
    min_x = 9999999
    min_y = 9999999
    max_x = 0
    max_y = 0

    for i in range(len(img)):
        for j in range(len(img[i])):
            if np.array_equal(img[i, j], [255, 255, 255]):
                if i < min_x:
                    min_x = i
                if i > max_x:
                    max_x = i
                if j < min_y:
                    min_y = j
                if j > max_y:
                    max_y = j

    cv2.rectangle(img, (min_y, min_x), (max_y, max_x), (0, 125, 0), 3)


if __name__ == '__main__':
    figures = {}

    # Test 1
    for file in os.listdir("data"):
        img = cv2.imread(f'data/{file}', -1)
        skin_map = np.zeros(img.shape)

        for i in range(len(img)):
            for j in range(len(img[i])):
                skin_map[i, j] = is_rgb_pixel_skin(img[i, j], mod=1)

        draw_face_shape(skin_map)
        figures[file + "_rgb1"] = skin_map

    # Test 2
    for file in os.listdir("data"):
        img = cv2.imread(f'data/{file}', -1)
        skin_map = np.zeros(img.shape)

        for i in range(len(img)):
            for j in range(len(img[i])):
                skin_map[i, j] = is_rgb_pixel_skin(img[i, j], mod=2)

        draw_face_shape(skin_map)
        figures[file + "_rgb2"] = skin_map

    # Test 3
    for file in os.listdir("data"):
        img = cv2.imread(f'data/{file}', -1)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        skin_map = np.zeros(hsv_img.shape)

        for i in range(len(hsv_img)):
            for j in range(len(hsv_img[i])):
                skin_map[i, j] = is_hsv_pixel_skin(hsv_img[i, j], mod=1)

        draw_face_shape(skin_map)
        figures[file + "_hsv1"] = skin_map

    # Test 4
    for file in os.listdir("data"):
        img = cv2.imread(f'data/{file}', -1)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        skin_map = np.zeros(hsv_img.shape)

        for i in range(len(hsv_img)):
            for j in range(len(hsv_img[i])):
                skin_map[i, j] = is_hsv_pixel_skin(hsv_img[i, j], mod=2)

        draw_face_shape(skin_map)
        figures[file + "_hsv2"] = skin_map

    # Test 5
    for file in os.listdir("data"):
        img = cv2.imread(f'data/{file}', -1)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        skin_map = np.zeros(hsv_img.shape)

        for i in range(len(hsv_img)):
            for j in range(len(hsv_img[i])):
                skin_map[i, j] = is_hsv_pixel_skin(hsv_img[i, j], mod=3)

        draw_face_shape(skin_map)
        figures[file + "_hsv3"] = skin_map

    # Test 6
    for file in os.listdir("data"):
        img = cv2.imread(f'data/{file}', -1)
        ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        skin_map = np.zeros(ycbcr_img.shape)

        for i in range(len(ycbcr_img)):
            for j in range(len(ycbcr_img[i])):
                skin_map[i, j] = is_ycbcr_pixel_skin(ycbcr_img[i, j], mod=1)

        draw_face_shape(skin_map)
        figures[file + "_ycrcb1"] = skin_map

    # Test 7
    for file in os.listdir("data"):
        img = cv2.imread(f'data/{file}', -1)
        ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        skin_map = np.zeros(ycbcr_img.shape)

        for i in range(len(ycbcr_img)):
            for j in range(len(ycbcr_img[i])):
                skin_map[i, j] = is_ycbcr_pixel_skin(ycbcr_img[i, j], mod=2)

        draw_face_shape(skin_map)
        figures[file + "_ycrcb2"] = skin_map

    plot_figures(figures, nrows=7, ncols=12)

    plt.show()
