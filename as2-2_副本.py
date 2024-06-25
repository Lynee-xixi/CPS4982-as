import cv2
import numpy as np
import matplotlib.pyplot as plt

# show BGR
image_path = 'Lena.jpg'
image = cv2.imread(image_path)
image_array = np.array(image)
height, width, channels = image_array.shape
image_rgb = np.zeros((height, width, channels), dtype=image_array.dtype)

image_rgb[:, :, 0] = image_array[:, :, 2]  # Red channel
image_rgb[:, :, 1] = image_array[:, :, 1]  # Green channel
image_rgb[:, :, 2] = image_array[:, :, 0]  # Blue channel

plt.imshow(image_rgb)
plt.title('Original')
plt.axis('off')
plt.show()


# change to grayscale image
def rgb_to_grayscale(rgb):
    grayscale = image_rgb[:, :, 0]*0.299 + image_rgb[:, :, 1]*0.587 + image_rgb[:, :, 2]*0.114
    return grayscale


lena_gray = rgb_to_grayscale(image_rgb)
plt.imshow(lena_gray, cmap='gray')  # 颜色变正常
plt.title('Grayscale Image')
plt.axis('off')
plt.show()


# 卷积应用sobel滤波器
def apply_sobel_filter(img, filter_kernel):
    # 显示过滤器属性
    print("Filter shape =", filter_kernel.shape)
    print(filter_kernel)

    # 根据过滤器尺寸计算输出图像的尺寸
    height_filter, width_filter = filter_kernel.shape
    h, w = img.shape
    output_h = h - height_filter + 1
    output_w = w - width_filter + 1

    # 初始化输出图像
    filtered_image = np.zeros((output_h, output_w))

    # 将过滤器应用于输入图像中的每个像素
    for i in range(output_h):
        for j in range(output_w):
            current_window = img[i:i + height_filter, j:j + width_filter]
            value = np.sum(current_window * filter_kernel)
            filtered_image[i, j] = value

    # 处理梯度值，将所有负值转换为正值
    filtered_image = np.abs(filtered_image)
    if filtered_image.max() != 0:
        filtered_image = 255 * (filtered_image / filtered_image.max())

    return filtered_image.astype(np.uint8)


# 定义索贝尔滤波器
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # X方向的梯度
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Y方向的梯度

# 应用索贝尔滤波器
gradient_x = apply_sobel_filter(lena_gray, sobel_x)
gradient_y = apply_sobel_filter(lena_gray, sobel_y)

# 显示梯度图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gradient_x, cmap='gray')
plt.title('Gradient X')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gradient_y, cmap='gray')
plt.title('Gradient Y')
plt.axis('off')

plt.show()


def calculate_gradient_magnitude(grad_x, grad_y):
    # 计算梯度幅值
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    # 归一化到0-255范围
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255
    return magnitude.astype(np.uint8)


# 计算梯度幅值
gradient_magnitude = calculate_gradient_magnitude(gradient_x, gradient_y)

# 显示梯度幅值图像
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')
plt.show()


# 将梯度幅值图像转换为二值图像
def convert_to_binary(img, threshold=128):
    binary_img = np.zeros_like(img)
    binary_img[img >= threshold] = 255
    return binary_img


# 选择阈值并生成二值图像
binary_edge_image = convert_to_binary(gradient_magnitude, threshold=128)

# 显示二值图像
plt.imshow(binary_edge_image, cmap='gray')
plt.title('Binary Edge Image')
plt.axis('off')
plt.show()

