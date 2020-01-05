import os

import cv2
import numpy as np
import time

directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
level = 2


def get_gclm_from_image(image, channel, direction, level=level):
    gclm_image = np.zeros((level, level, 1))

    start_i = 0 if direction[0] != -1 else 1
    end_i = len(image) if direction[0] != 1 else len(image) - 1
    start_j = 0 if direction[1] != -1 else 1
    end_j = len(image[0]) if direction[1] != 1 else len(image[0]) - 1

    for i in range(start_i, end_i):
        for j in range(start_j, end_j):
            first_pixel = int(image[i, j, channel] / 256 * level)
            second_pixel = int(image[i + direction[0], j + direction[1], channel] / 256 * level)

            gclm_image[first_pixel, second_pixel] += 1 / level ** 2

    return gclm_image


def get_features_from_gclm(gclm, level=level):
    # compute mr and mc
    mr = 0
    mc = 0
    for i in range(level):
        sum_r = 0
        sum_c = 0

        for j in range(level):
            sum_r += gclm[i, j]
            sum_c += gclm[j, i]

        mr += i * sum_r
        mc += i * sum_c

    # compute sigma_r and sigma_c
    sigma_r = 0
    sigma_c = 0

    for i in range(level):
        sum_r = 0
        sum_c = 0

        for j in range(level):
            sum_r += gclm[i, j]
            sum_c += gclm[j, i]

        sigma_r += (i - mr) * sum_r
        sigma_c += (i - mc) * sum_c

    correlation = 0
    contrast = 0
    uniformity = 0
    homogenity = 0
    entropy = 0
    dissimilarity = 0
    for i in range(level):
        for j in range(level):
            correlation += (i - mr) * (j - mc) * gclm[i, j] / (sigma_c * sigma_r)
            contrast += (i - j) ** 2 * gclm[i, j]
            uniformity += gclm[i, j] ** 2
            homogenity += gclm[i, j] / (1 + np.abs(i - j))
            entropy -= gclm[i, j] * np.where(gclm[i, j] > 0.0000000001, gclm[i, j], -10)
            dissimilarity += np.abs(i - j) * gclm[i, j]

    return (
        np.max(gclm),
        correlation,
        contrast,
        uniformity,
        homogenity,
        entropy,
        dissimilarity
    )


# noinspection PyShadowingNames
def extract_features(x):
    image = x[0]
    target = x[1]

    print(target)
    image_feature = np.zeros((3, len(directions), 7))

    # for each color
    for color in range(3):
        # for each direction
        for idx, direction in enumerate(directions):
            print("getting gclm...")
            start = time.time()
            gclm = get_gclm_from_image(image=image, channel=color, direction=direction)
            end = time.time()
            print(f"got gclm... {end - start}")
            start = time.time()
            image_feature[color, idx] = get_features_from_gclm(gclm)
            end = time.time()
            print(f"got features... {end - start}")

    return image_feature, target


if __name__ == '__main__':
    files = os.listdir("data")

    iris_data_set = []
    for idx, image_name in enumerate(files):
        img = cv2.imread(f'data/{image_name}', 1)
        iris_data_set.append((img, idx // 6))

    features = map(extract_features, iris_data_set)
    print(list(features))
