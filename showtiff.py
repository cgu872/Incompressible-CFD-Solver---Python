# -*- coding:utf-8 -*-
# @Time     :2024/9/17

import numpy as np
import pandas as pd
from pyproj import Proj
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import cv2


if __name__ == '__main__':
    path=r'C:\\Users\\cgu872\\Downloads\\ASTGTMV003_S37E174_dem.tif'
    img_tf=tf.imread(path)
    # plt.imshow(img_tf,cmap='gray')
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()
    # img = Image.open(path)
    # img.show()
    # cv2.imshow('Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 读取 TIFF 图像
    image = tf.imread(path)
    # 创建网格
    x = np.linspace(0, image.shape[1], image.shape[1])
    y = np.linspace(0, image.shape[0], image.shape[0])
    x, y = np.meshgrid(x, y)
    # 创建图形和三维轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制三维表面图
    ax.plot_surface(x, y, image, cmap='terrain')
    # 添加标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Elevation')
    # 显示图形
    plt.show()

