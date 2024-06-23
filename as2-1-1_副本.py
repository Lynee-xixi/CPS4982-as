import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image_path = 'koala.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 确保是以灰度模式读取


# Salt noise:
# randomly setting some pixel values in the image to the maximum value(white pixels)
def add_salt_noise(img, noise_ratio):
    noisy_img1 = np.copy(img)  # 必须copy，以确保原图不会改变
    # 计算被改变的pixel的数量
    num_noise = int(noise_ratio * img.size)
    # 随机选择需要改变的像素位置
    x_indices = np.random.randint(0, img.shape[0], num_noise)
    y_indices = np.random.randint(0, img.shape[1], num_noise)
    # 将随机选择的像素设置为最大值（灰度值为 255）
    noisy_img1[x_indices, y_indices] = 255
    return noisy_img1


# 给图像添加salt noise
salt_noisy_image = add_salt_noise(image, 0.07)


# Pepper noise:
# randomly setting some pixel values in the image to the minimum value(black pixels)
# 和salt相似，但改变这些随机的pixel为黑色，也就是灰度值为0
def add_pepper_noise(img, noise_ratio):
    noisy_img2 = np.copy(img)
    num_noise = int(noise_ratio * image.size)
    x_indices = np.random.randint(0, img.shape[0], num_noise)
    y_indices = np.random.randint(0, img.shape[1], num_noise)
    # 将随机选择的像素设置为最小值（灰度值为 0）
    noisy_img2[x_indices, y_indices] = 0
    return noisy_img2


# 给图像添加pepper noise
pepper_noisy_image = add_pepper_noise(image, 0.07)


# Gaussian noise:
# affects each pixel in the image by adding a random value from theGaussian distribution to the original pixel value
def add_gaussian_noise(img, mean=0, std=20):
    # 生成Gaussian noise
    gaussian_noise = np.random.normal(mean, std, img.shape)
    # 将Gaussian noise添加到图像中
    noisy_img3 = img + gaussian_noise
    # 将值限制在0到255范围内
    noisy_img3 = noisy_img3.astype(np.float32)  # 确保在操作前转换类型以避免溢出
    noisy_img3[noisy_img3 < 0] = 0   # 小于0的值设置为0
    noisy_img3[noisy_img3 > 255] = 255  # 大于255的值设置为255
    # 将浮点数转换为uint8
    noisy_img3 = noisy_img3.astype(np.uint8)
    return noisy_img3


# 向图像添加Gaussian noise
gaussian_noisy_image = add_gaussian_noise(image)

# 可视化部分
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 5))

ax1.imshow(image, cmap='gray', vmin=0, vmax=255)
ax1.set_title('Original Image')
ax1.axis('off')

ax2.imshow(salt_noisy_image, cmap='gray', vmin=0, vmax=255)
ax2.set_title('Salt noise')
ax2.axis('off')

ax3.imshow(pepper_noisy_image, cmap='gray', vmin=0, vmax=255)
ax3.set_title('Pepper noise')
ax3.axis('off')

ax4.imshow(gaussian_noisy_image, cmap='gray', vmin=0, vmax=255)
ax4.set_title('Gaussian noise')
ax4.axis('off')

plt.show()


# 定义保存图片的函数，这个部分是gpt的用于导出这几张生成的图片，用于第二题
def save_image_if_not_exists(img, filename):
    # 检查文件是否已存在
    if not os.path.exists(filename):
        # 如果不存在，保存图片
        cv2.imwrite(filename, img)
        print(f"Image saved as {filename}")
    else:
        # 如果存在，不保存图片
        print(f"File {filename} already exists. No new file saved.")


# 使用定义的函数保存图像
save_image_if_not_exists(pepper_noisy_image, "pepper_noisy_image.jpg")
save_image_if_not_exists(salt_noisy_image, "salt_noisy_image.jpg")
save_image_if_not_exists(gaussian_noisy_image, "gaussian_noisy_image.jpg")

