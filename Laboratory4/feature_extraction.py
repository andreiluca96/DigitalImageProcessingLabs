import os

import cv2
import numpy as np

directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
level = 256


def get_gclm_from_image(image, direction, level=level):
    gclm_image = np.zeros((level, level, 3))

    start_i = 0 if direction[0] != -1 else 1
    end_i = len(image) if direction[0] != 1 else len(image) - 1
    start_j = 0 if direction[1] != -1 else 1
    end_j = len(image[0]) if direction[1] != 1 else len(image[0]) - 1

    for i in range(start_i, end_i):
        for j in range(start_j, end_j):
            for channel in range(3):
                first_pixel = int(image[i, j, channel] / 256 * level)
                second_pixel = int(image[i + direction[0], j + direction[1], channel] / 256 * level)

                gclm_image[first_pixel, second_pixel, channel] += 1 / level ** 2

    return gclm_image


def get_features_from_gclm(gclm, level=level):
    # compute mr and mc
    mr = [0, 0, 0]
    mc = [0, 0, 0]
    for i in range(level):
        sum_r = [0, 0, 0]
        sum_c = [0, 0, 0]

        for j in range(level):
            for c in range(3):
                sum_r[c] += gclm[i, j, c]
                sum_c[c] += gclm[j, i, c]

        for c in range(3):
            mr[c] += i * sum_r[c]
            mc[c] += i * sum_c[c]

    # compute sigma_r and sigma_c
    sigma_r = [0, 0, 0]
    sigma_c = [0, 0, 0]

    for i in range(level):
        sum_r = [0, 0, 0]
        sum_c = [0, 0, 0]

        for j in range(level):
            for c in range(3):
                sum_r[c] += gclm[i, j, c]
                sum_c[c] += gclm[j, i, c]
        for c in range(3):
            sigma_r[c] += (i - mr[c]) * sum_r[c]
            sigma_c[c] += (i - mc[c]) * sum_c[c]

    correlation = [0, 0, 0]
    contrast = [0, 0, 0]
    uniformity = [0, 0, 0]
    homogenity = [0, 0, 0]
    entropy = [0, 0, 0]
    dissimilarity = [0, 0, 0]
    for i in range(level):
        for j in range(level):
            for c in range(3):
                correlation[c] += (i - mr[c]) * (j - mc[c]) * gclm[i, j, c] / (sigma_c[c] * sigma_r[c] + 0.001)
                contrast[c] += (i - j) ** 2 * gclm[i, j, c]
                uniformity[c] += gclm[i, j, c] ** 2
                homogenity[c] += gclm[i, j, c] / (1 + np.abs(i - j))
                entropy[c] -= gclm[i, j, c] * np.where(gclm[i, j, c] > 0.0000000001, gclm[i, j, c], -10)
                dissimilarity[c] += np.abs(i - j) * gclm[i, j, c]

    return (
        np.max(gclm),
        correlation[0],
        correlation[1],
        correlation[2],
        contrast[0],
        contrast[1],
        contrast[2],
        uniformity[0],
        uniformity[1],
        uniformity[2],
        homogenity[0],
        homogenity[1],
        homogenity[2],
        entropy[0],
        entropy[1],
        entropy[2],
        dissimilarity[0],
        dissimilarity[1],
        dissimilarity[2]
    )


# noinspection PyShadowingNames
def extract_features(image):
    image_feature = np.zeros((len(directions) * 19))

    # for each direction
    for idx, direction in enumerate(directions):
        gclm = get_gclm_from_image(image=image, direction=direction)
        image_feature[idx * 19:(idx + 1) * 19] = get_features_from_gclm(gclm)
    return image_feature


if __name__ == '__main__':
    files = os.listdir("data")
    my_idx = 0
    iris_data_set = []
    targets = []
    for idx, image_name in enumerate(files):
        img = cv2.imread(f'data/{image_name}', 1)
        iris_data_set.append(img)
        targets.append(idx // 6)

    counter = 0
    features = []
    for entry in iris_data_set:
        counter += 1
        print(counter)
        features.append(extract_features(entry))
