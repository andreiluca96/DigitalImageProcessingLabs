import numpy as np

from utils.dip_utils import plot_figures
from matplotlib import pyplot as plt
import cv2


def gray_scale_simple_averaging_simple(image):
    gray_scale_image = np.zeros(img.shape[:-1], dtype=int)

    for i in range(len(image)):
        for j in range(len(image[i])):
            pixel = image[i, j]

            r = int(pixel[0])
            g = int(pixel[1])
            b = int(pixel[2])

            average = round((r + g + b) / 3)
            gray_scale_image[i, j] = average

    return gray_scale_image


def gray_scale_simple_averaging(image):
    gray_scale_image = np.zeros(img.shape, dtype=int)

    for i in range(len(image)):
        for j in range(len(image[i])):
            pixel = image[i, j]

            r = int(pixel[0])
            g = int(pixel[1])
            b = int(pixel[2])

            average = round((r + g + b) / 3)
            gray_scale_image[i, j] = [average, average, average]

    return gray_scale_image


def gray_scale_weighted_average(image):
    gray_scale_image = np.zeros(img.shape, dtype=int)

    for i in range(len(image)):
        for j in range(len(image[i])):
            pixel = image[i, j]

            r = int(pixel[0])
            g = int(pixel[1])
            b = int(pixel[2])

            weighed_average = round(0.3 * r + 0.59 * g + 0.11 * b)
            gray_scale_image[i, j] = [weighed_average, weighed_average, weighed_average]

    return gray_scale_image


def gray_scale_desaturation(image):
    gray_scale_image = np.zeros(img.shape, dtype=int)

    for i in range(len(image)):
        for j in range(len(image[i])):
            pixel = image[i, j]
            gray_value = round((int(min(pixel)) + int(max(pixel))) / 2)
            gray_scale_image[i, j] = [gray_value, gray_value, gray_value]

    return gray_scale_image


def gray_scale_decomposition(image, mod='max'):
    gray_scale_image = np.zeros(img.shape, dtype=int)

    if mod == 'max':
        for i in range(len(image)):
            for j in range(len(image[i])):
                pixel = image[i, j]
                gray_value = round(max(pixel))
                gray_scale_image[i, j] = [gray_value, gray_value, gray_value]

    if mod == 'min':
        for i in range(len(image)):
            for j in range(len(image[i])):
                pixel = image[i, j]
                gray_value = round(min(pixel))
                gray_scale_image[i, j] = [gray_value, gray_value, gray_value]

    return gray_scale_image


def gray_scale_single_channel(image, mod='r'):
    gray_scale_image = np.zeros(img.shape, dtype=int)

    if mod == 'r':
        for i in range(len(image)):
            for j in range(len(image[i])):
                pixel = image[i, j]
                gray_value = pixel[0]
                gray_scale_image[i, j] = [gray_value, gray_value, gray_value]

    if mod == 'g':
        for i in range(len(image)):
            for j in range(len(image[i])):
                pixel = image[i, j]
                gray_value = pixel[1]
                gray_scale_image[i, j] = [gray_value, gray_value, gray_value]

    if mod == 'b':
        for i in range(len(image)):
            for j in range(len(image[i])):
                pixel = image[i, j]
                gray_value = pixel[2]
                gray_scale_image[i, j] = [gray_value, gray_value, gray_value]

    return gray_scale_image


def gray_scale_custom_gray_shade(image, shades):
    gray_scale_image = np.zeros(img.shape, dtype=int)

    for i in range(len(image)):
        for j in range(len(image[i])):
            pixel = image[i, j]

            r = int(pixel[0])
            g = int(pixel[1])
            b = int(pixel[2])

            weighed_average = round(0.3 * r + 0.59 * g + 0.11 * b)

            for shade in shades:
                if shade[0] <= weighed_average <= shade[1]:
                    weighed_average = round((shade[0] + shade[1]) / 2)

            gray_scale_image[i, j] = [weighed_average, weighed_average, weighed_average]

    return gray_scale_image


def gray_scale_custom_number_grey_shades_with_error_diffusion(image, shades):
    gray_scale_image = np.zeros(img.shape, dtype=int)

    for i in range(len(image)):
        for j in range(len(image[i])):
            pixel = image[i, j]

            r = int(pixel[0])
            g = int(pixel[1])
            b = int(pixel[2])

            weighed_average = round(0.3 * r + 0.59 * g + 0.11 * b)

            for shade in shades:
                if shade[0] <= weighed_average <= shade[1]:
                    if weighed_average - shade[0] > shade[1] - weighed_average:
                        gray_value = shade[1]
                    else:
                        gray_value = shade[0]

            error = weighed_average - gray_value
            if i < len(image) - 1 and j < len(image) - 1:
                image[i, j + 1] = [int(image[i, j + 1, 0]) + round(error * 7 / 16),
                                   int(image[i, j + 1, 1]) + round(error * 7 / 16),
                                   int(image[i, j + 1, 2]) + round(error * 7 / 16)]
                image[i + 1, j - 1] = [int(image[i + 1, j - 1, 0]) + round(error * 3 / 16),
                                       int(image[i + 1, j - 1, 1]) + round(error * 3 / 16),
                                       int(image[i + 1, j - 1, 2]) + round(error * 3 / 16)]
                image[i + 1, j] = [int(image[i + 1, j, 0]) + round(error * 5 / 16),
                                   int(image[i + 1, j, 1]) + round(error * 5 / 16),
                                   int(image[i + 1, j, 2]) + round(error * 5 / 16)]
                image[i + 1, j + 1] = [int(image[i + 1, j + 1, 0]) + round(error * 1 / 16),
                                       int(image[i + 1, j + 1, 1]) + round(error * 1 / 16),
                                       int(image[i + 1, j + 1, 2]) + round(error * 1 / 16)]

            gray_scale_image[i, j] = [gray_value, gray_value, gray_value]

    return gray_scale_image


if __name__ == '__main__':
    figures = {}

    # original figure
    img = cv2.imread(f'data/example.jpg', 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    figures['original'] = rgb_img

    figures['simple_averaging'] = gray_scale_simple_averaging(rgb_img)
    figures['weighted_average'] = gray_scale_weighted_average(rgb_img)
    figures['desaturation'] = gray_scale_desaturation(rgb_img)

    figures['decomposition_max'] = gray_scale_decomposition(rgb_img, mod='max')
    figures['decomposition_min'] = gray_scale_decomposition(rgb_img, mod='min')

    figures['single_channel_r'] = gray_scale_single_channel(rgb_img, mod='r')
    figures['single_channel_g'] = gray_scale_single_channel(rgb_img, mod='g')
    figures['single_channel_b'] = gray_scale_single_channel(rgb_img, mod='b')

    figures['custom_gray_shade3intervals'] = gray_scale_custom_gray_shade(rgb_img, [(0, 100), (100, 200), (200, 255)])
    figures['custom_gray_shade5intervals'] = gray_scale_custom_gray_shade(rgb_img,
                                                                          [(0, 50), (50, 100), (100, 150), (150, 200),
                                                                           (200, 255)])
    figures['custom_gray_shade7intervals'] = gray_scale_custom_gray_shade(rgb_img,
                                                                          [(0, 50), (50, 100), (100, 150), (150, 175),
                                                                           (175, 200), (200, 225), (225, 255)])
    figures['diffusion3intervals'] = gray_scale_custom_number_grey_shades_with_error_diffusion(rgb_img.copy(),
                                                                                               [(0, 100), (100, 200),
                                                                                                (200, 255)])
    figures['diffusion5intervals'] = gray_scale_custom_number_grey_shades_with_error_diffusion(rgb_img.copy(),
                                                                                               [(0, 50), (50, 100),
                                                                                                (100, 150), (150, 200),
                                                                                                (200, 255)])
    figures['diffusion7intervals'] = gray_scale_custom_number_grey_shades_with_error_diffusion(rgb_img.copy(),
                                                                                               [(0, 50), (50, 100),
                                                                                                (100, 150), (150, 175),
                                                                                                (175, 200), (200, 225),
                                                                                                (225, 255)])

    # plot results
    plot_figures(figures, nrows=5, ncols=3)
    plt.show()
