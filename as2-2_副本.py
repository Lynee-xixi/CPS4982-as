import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载灰度图像
image_path = 'Assignment-1-2-HE-source.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 展示原始图像
plt.imshow(image, cmap='gray')
plt.title('Original Image')
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


# 累积分布函数（CDF）
def calculate_cdf(hist):
    cdf = np.zeros_like(hist)
    cumulative_sum = 0
    for i in range(len(hist)):
        cumulative_sum += hist[i]
        cdf[i] = cumulative_sum
    return cdf


def histogram_equalization(img, hist):
    cdf = calculate_cdf(hist)
    # 忽略零并找到最小非零值（从as1-1的直方图发现从0灰度级开始像素数为0的有很大一部分）
    cdf_min = cdf[np.nonzero(cdf)[0][0]]
    cdf_normalized = (cdf - cdf_min) / (img.size - cdf_min)  # 归一化 CDF

    # 根据CDF映射旧像素值到新值
    img_eq = np.interp(img.flatten(), range(256),
                       cdf_normalized * 255).astype('uint8')
    img_eq = np.reshape(img_eq, img.shape)

    return img_eq


# 计算直方图
histogram = calculate_histogram(image)

# 执行直方图均衡化
image_eq = histogram_equalization(image, histogram)

# 计算均衡化后的直方图
histogram_eq = calculate_histogram(image_eq)

# 展示均衡化后的图像
plt.imshow(image_eq, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')
plt.show()

# 绘制原始与均衡化后的直方图
plt.bar(range(256), histogram, color='gray')
plt.title('Original Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

plt.bar(range(256), histogram_eq, color='gray')
plt.title('Equalized Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()
