import numpy as np
import matplotlib.pyplot as plt

def simulate_projections(num_projections, num_samples):
    """模拟投影数据"""
    angles = np.linspace(0, np.pi, num_projections)  # 选择角度
    projections = np.zeros((num_projections, num_samples))

    # 模拟一个圆形物体的投影
    for i, angle in enumerate(angles):
        x = np.linspace(-1, 1, num_samples)
        projections[i] = np.maximum(0, 1 - np.abs(x * np.cos(angle) + x * np.sin(angle)))

    return projections

num_projections = 360  # 投影条数
num_samples = 100      # 每个投影样本数
projections = simulate_projections(num_projections, num_samples)

# 绘制投影图
plt.imshow(projections, extent=(0, num_samples, 0, num_projections), aspect='auto')
plt.title("Simulated Projections")
plt.xlabel("Sample")
plt.ylabel("Projection Angle")
plt.colorbar()
plt.show()

def ram_lak_filter(num_samples):
    """创建Ram-Lak滤波器"""
    freq = np.fft.fftfreq(num_samples)
    filter = 2 * np.abs(freq)
    filter[freq < 0] = 0
    return filter

def filter_projections(projections):
    """对投影数据应用滤波器"""
    num_samples = projections.shape[1]
    filter = ram_lak_filter(num_samples)
    
    filtered_projections = np.zeros_like(projections)
    for i in range(projections.shape[0]):
        P = np.fft.fft(projections[i])
        filtered_projections[i] = np.fft.ifft(P * filter).real

    return filtered_projections

filtered_projections = filter_projections(projections)

# 绘制滤波后的投影图
plt.imshow(filtered_projections, extent=(0, num_samples, 0, num_projections), aspect='auto')
plt.title("Filtered Projections")
plt.xlabel("Sample")
plt.ylabel("Projection Angle")
plt.colorbar()
plt.show()


def back_projection(filtered_projections):
    """反投影过滤后的数据"""
    num_projections, num_samples = filtered_projections.shape
    image = np.zeros((num_samples, num_samples))  # 创建重建图像
    
    for i in range(num_projections):
        angle = i * np.pi / num_projections
        for x in range(num_samples):
            for y in range(num_samples):
                # 将极坐标转换为笛卡尔坐标
                proj_x = x * np.cos(angle) + y * np.sin(angle)
                if 0 <= int(proj_x) < num_samples:
                    image[y, x] += filtered_projections[i, int(proj_x)]
    
    return image

rebuilt_image = back_projection(filtered_projections)

# 绘制重建图像
plt.imshow(rebuilt_image, cmap='gray')
plt.title("Reconstructed Image")
plt.colorbar()
plt.show()

def normalize_image(image):
    """归一化重建图像"""
    return (image - np.min(image)) / (np.max(image) - np.min(image))

normalized_image = normalize_image(rebuilt_image)

# 绘制归一化后的图像
plt.imshow(normalized_image, cmap='gray')
plt.title("Normalized Reconstructed Image")
plt.colorbar()
plt.show()