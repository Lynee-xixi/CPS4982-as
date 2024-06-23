import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image_path = 'salt_noisy_image.jpg'
salt_noisy_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 确保是以灰度模式读取
image_path = 'pepper_noisy_image.jpg'
pepper_noisy_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_path = 'gaussian_noisy_image.jpg'
gaussian_noisy_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_path = 'koala.jpg'  # 原图，对比用
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def median_filter(img, filter_size=3):
    h, w = img.shape
    k = filter_size // 2
    output_h = h - 2 * k
    output_w = w - 2 * k
    filtered_image = np.zeros((output_h, output_w), dtype=img.dtype)

    for i in range(k, h-k):
        for j in range(k, w-k):
            current_window = img[i-k:i+k+1, j-k:j+k+1]
            sorted_window = np.sort(current_window, axis=None)
            median_value = sorted_window[len(sorted_window) // 2]
            filtered_image[i-k, j-k] = median_value

    return filtered_image


def maximum_filter(img, filter_size=3):
    h, w = img.shape
    k = filter_size // 2
    output_h = h - 2 * k
    output_w = w - 2 * k
    filtered_image = np.zeros((output_h, output_w), dtype=img.dtype)

    for i in range(k, h-k):
        for j in range(k, w-k):
            current_window = img[i-k:i+k+1, j-k:j+k+1]
            sorted_window = np.sort(current_window, axis=None)
            max_value = sorted_window[-1]
            filtered_image[i-k, j-k] = max_value

    return filtered_image


def minimum_filter(img, filter_size=3):
    h, w = img.shape
    k = filter_size // 2
    output_h = h - 2 * k
    output_w = w - 2 * k
    filtered_image = np.zeros((output_h, output_w), dtype=img.dtype)

    for i in range(k, h-k):
        for j in range(k, w-k):
            current_window = img[i-k:i+k+1, j-k:j+k+1]
            sorted_window = np.sort(current_window, axis=None)
            min_value = sorted_window[0]
            filtered_image[i-k, j-k] = min_value

    return filtered_image


# 应用filter
median_filtered_salt = median_filter(salt_noisy_image, 3)
median_filtered_pepper = median_filter(pepper_noisy_image, 3)
median_filtered_gaussian = median_filter(gaussian_noisy_image, 3)
maximum_filtered_salt = maximum_filter(salt_noisy_image, 3)
maximum_filtered_pepper = maximum_filter(pepper_noisy_image, 3)
maximum_filtered_gaussian = maximum_filter(gaussian_noisy_image, 3)
minimum_filtered_salt = minimum_filter(salt_noisy_image, 3)
minimum_filtered_pepper = minimum_filter(pepper_noisy_image, 3)
minimum_filtered_gaussian = minimum_filter(gaussian_noisy_image, 3)


fig, axes = plt.subplots(4, 4, figsize=(13, 9))  # 调整布局以显示三种类型的结果

# 显示原始的noise的图像
axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')
axes[0, 1].imshow(salt_noisy_image, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('salt_noisy_image')
axes[0, 1].axis('off')
axes[0, 2].imshow(pepper_noisy_image, cmap='gray', vmin=0, vmax=255)
axes[0, 2].set_title('pepper_noisy_image')
axes[0, 2].axis('off')
axes[0, 3].imshow(gaussian_noisy_image, cmap='gray', vmin=0, vmax=255)
axes[0, 3].set_title('gaussian_noisy_image')
axes[0, 3].axis('off')

# 显示中值滤波后的图像
axes[1, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
axes[1, 0].set_title('Original Image')
axes[1, 0].axis('off')
axes[1, 1].imshow(median_filtered_salt, cmap='gray', vmin=0, vmax=255)
axes[1, 1].set_title('Median Salt')
axes[1, 1].axis('off')
axes[1, 2].imshow(median_filtered_pepper, cmap='gray', vmin=0, vmax=255)
axes[1, 2].set_title('Median Pepper')
axes[1, 2].axis('off')
axes[1, 3].imshow(median_filtered_gaussian, cmap='gray', vmin=0, vmax=255)
axes[1, 3].set_title('Median Gaussian')
axes[1, 3].axis('off')

# 显示最大值滤波后的图像
axes[2, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
axes[2, 0].set_title('Original Image')
axes[2, 0].axis('off')
axes[2, 1].imshow(maximum_filtered_salt, cmap='gray', vmin=0, vmax=255)
axes[2, 1].set_title('Maximum Salt')
axes[2, 1].axis('off')
axes[2, 2].imshow(maximum_filtered_pepper, cmap='gray', vmin=0, vmax=255)
axes[2, 2].set_title('Maximum Pepper')
axes[2, 2].axis('off')
axes[2, 3].imshow(maximum_filtered_gaussian, cmap='gray', vmin=0, vmax=255)
axes[2, 3].set_title('Maximum Gaussian')
axes[2, 3].axis('off')

# 显示最小值滤波后的图像
axes[3, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
axes[3, 0].set_title('Original Image')
axes[3, 0].axis('off')
axes[3, 1].imshow(minimum_filtered_salt, cmap='gray', vmin=0, vmax=255)
axes[3, 1].set_title('Minimum Salt')
axes[3, 1].axis('off')
axes[3, 2].imshow(minimum_filtered_pepper, cmap='gray', vmin=0, vmax=255)
axes[3, 2].set_title('Minimum Pepper')
axes[3, 2].axis('off')
axes[3, 3].imshow(minimum_filtered_gaussian, cmap='gray', vmin=0, vmax=255)
axes[3, 3].set_title('Minimum Gaussian')
axes[3, 3].axis('off')

plt.tight_layout()
plt.show()

