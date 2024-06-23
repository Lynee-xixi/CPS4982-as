import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image_path = 'Assignment-1-2-HE-source.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 确保是以灰度模式读取

# 显示原始灰度图
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis('off')
plt.show()


def calculate_histogram(img):
    # 初始化直方图数组，256个灰度级
    hist = np.zeros(256, dtype=int)

    # 统计每个灰度级的像素数
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            pixel_value = img[i, j]
            hist[pixel_value] += 1

    return hist


# 计算直方图
histogram = calculate_histogram(image)
total_pixels = image.shape[0] * image.shape[1]
probability_histogram = histogram / total_pixels

# 显示频率直方图
plt.bar(range(256), histogram, color='gray')
plt.title('Frequency Histogram of the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

# 显示概率直方图
plt.bar(range(256), probability_histogram, color='gray')
plt.title('Probability Histogram of the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability')
plt.show()


