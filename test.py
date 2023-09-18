import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import torch.nn as nn
from train import WavefunctionPredictor

# 定义画布的宽度和高度
WIDTH = 800
HEIGHT = 600

# 定义势能和波函数的坐标范围
POTENTIAL_MIN = 0
POTENTIAL_MAX = HEIGHT
WAVEFUNCTION_MIN = 0
WAVEFUNCTION_MAX = 1

# 初始化pygame
pygame.init()

# 创建一个窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand-drawn Potential")

# 用于存储绘制的势能数据
potential = []

# 加载模型
model = WavefunctionPredictor(HEIGHT, HEIGHT)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# 游戏循环
running = True
while running:
    # 清空屏幕
    screen.fill((255, 255, 255))

    # 处理事件
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEBUTTONDOWN:
            # 鼠标左键按下，记录当前点的坐标
            if event.button == 1:
                pos = pygame.mouse.get_pos()
                potential.append(pos[1])
        elif event.type == KEYDOWN:
            # 按下空格键进行预测并保存图片
            if event.key == K_SPACE:
                # 转换势能数据的坐标范围
                potential_normalized = np.interp(potential, (POTENTIAL_MIN, POTENTIAL_MAX), (WAVEFUNCTION_MIN, WAVEFUNCTION_MAX))

                # 转换势能数据为PyTorch的Tensor类型
                potential_tensor = torch.Tensor(potential_normalized).unsqueeze(0)

                # 使用模型进行预测
                with torch.no_grad():
                    predicted_wavefunction = model(potential_tensor)

                # 将预测的波函数数据转换为NumPy数组
                predicted_wavefunction = predicted_wavefunction.numpy().squeeze()

                # 可视化绘制势能和预测的波函数
                fig = plt.figure(figsize=(8, 6))
                gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5])

                ax1 = plt.subplot(gs[0])
                ax1.plot(potential_normalized, label='Potential')
                ax1.set_title('Potential')

                ax2 = plt.subplot(gs[1])
                ax2.plot(predicted_wavefunction ** 2, label='Predicted Wavefunction')
                ax2.set_title('Wavefunction')

                ax1.legend(loc='upper right')
                ax2.legend(loc='upper right')

                plt.tight_layout(h_pad=1.5)
                plt.savefig('predicted_wavefunction.png')
                plt.close()

    # 绘制势能
    if len(potential) > 1:
        pygame.draw.lines(screen, (0, 0, 0), False, [(i, potential[i]) for i in range(len(potential))], 2)

    # 更新屏幕
    pygame.display.update()

# 退出pygame
pygame.quit()
