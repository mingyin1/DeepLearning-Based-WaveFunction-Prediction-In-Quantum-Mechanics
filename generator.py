import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from matplotlib import gridspec
import shutil
import random
import os
import threading
import argparse
import sympy
import scipy

parser = argparse.ArgumentParser(description='Matrix Factorization')
parser.add_argument('--num', type=int ,default=10000)
parser.add_argument('--dx', type=float ,default=0.2)
parser.add_argument('--mass', type = float,default=1.0)

args = parser.parse_args()
#生成服从子指数分布的随机数
def subexp(expon):
    return np.power(abs(np.log(np.random.uniform())), expon)

# 生成随机势能
def generate_random_potential(length, low, high):
    return np.random.uniform(low, high, length)

#生成阶梯状的势能
def generate_step_potential(length, step_height):
    num_steps = length // step_height
    potential = np.repeat(np.arange(num_steps), step_height)
    remaining_length = length - len(potential)
    if remaining_length > 0:
        potential = np.concatenate((potential, np.repeat(num_steps, remaining_length)))
    return potential

#生成锯齿状的势能
def generate_shape_potential(length, pattern_length):
    num_patterns = length // pattern_length
    potential = np.tile(np.arange(pattern_length), num_patterns)
    remaining_length = length - len(potential)
    if remaining_length > 0:
        potential = np.concatenate((potential, np.arange(remaining_length)))
    return potential

def calculate_wavefunction(potential, dx, mass):
    """
    计算一维势场中的定态波函数

    参数：
    potential: array-like, 势能数组
    dx: float, 空间步长
    mass: float, 粒子的质量（默认为1.0）

    返回值：
    wave_function: array-like, 定态波函数
    """

    # 计算势能矩阵
    diagonal = 2.0 * mass / dx**2 + potential
    off_diagonal = -mass / dx**2 * np.ones(len(potential) - 1)
    potential_matrix = np.diag(diagonal) + np.diag(off_diagonal, k=1) + np.diag(off_diagonal, k=-1)

    # 求解薛定谔方程的本征值和本征函数
    eigenvalues, eigenvectors = eigh_tridiagonal(diagonal, off_diagonal)

    # 找到最低的一个能级
    num_levels = min(1, len(eigenvalues))
    lowest_eigenvalues = eigenvalues[:num_levels]
    lowest_eigenvectors = eigenvectors[:, :num_levels]

    # 归一化波函数
    norm = np.sqrt(np.sum(lowest_eigenvectors**2, axis=0))
    normalized_eigenvectors = lowest_eigenvectors / norm
    return normalized_eigenvectors


# 生成训练和验证数据
def generate_data(num_samples, train_ratio):
    potential_data = []
    wavefunction_data = []

    for i in range(num_samples):
        random_integer = random.randint(0, 5)
        if random_integer >=0 and  random_integer <= 3:     #如果随机数为0-3之间则生成完全随机势能
            potential = generate_random_potential(length,low,high)
        elif  random_integer ==4:           #如果随机数为4则生成阶梯状势能
            step_height = random.randint(2,20)
            potential = generate_step_potential(length,step_height)
        elif random_integer == 5:           #如果随机数为5则生成锯齿状势能
            pattern_length = random.randint(1, 12)
            potential = generate_shape_potential(length,pattern_length)

        wavefunction = calculate_wavefunction(potential,dx,mass)
        potential_data.append(potential)
        wavefunction_data.append(wavefunction)
        if i%100 ==0:
            print(f"Already generated:{i}.")

    # 划分训练和验证数据
    train_samples = int(num_samples * train_ratio)
    train_potential = np.array(potential_data[:train_samples])
    train_wavefunction = np.array(wavefunction_data[:train_samples])
    test_potential = np.array(potential_data[train_samples:])
    test_wavefunction = np.array(wavefunction_data[train_samples:])

    # 保存数据
    np.save("data/train_potential.npy", train_potential)           #将势能数据保存到npy文件中
    np.save("data/train_wavefunction.npy", train_wavefunction)     ##将计算出来的波函数数据保存到npy文件中
    np.save("data/test_potential.npy", test_potential)
    np.save("data/test_wavefunction.npy", test_wavefunction)

    return train_potential, train_wavefunction, test_potential, test_wavefunction

# 示例参数
num_samples = args.num  # 总样本数量
train_ratio = 0.8  # 训练数据所占比例
length = 100  # 势能数组的长度
dx = args.dx      #步长
mass = args.mass    #粒子质量
low = 0.1     #随机生成势能的最小值
high = 1.0    #随机生成势能的最大值
# 生成数据
train_potential, train_wavefunction, test_potential, test_wavefunction = generate_data(num_samples, train_ratio)

# 可视化势能和波函数，并保存为多张图片

if os.path.exists('pic_of_wavefunction'):
    shutil.rmtree('pic_of_wavefunction')
os.makedirs('pic_of_wavefunction')
print(f"Directory pic_of_wavefunction created!")

def generate_image(i):
    fig = plt.figure(figsize=(8, 6))                 
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5])   # 创建包含两个子图的画布
    
    ax1 = plt.subplot(gs[0])
    ax1.plot(train_potential[i], label='Potential')   # 在第一个子图中绘制势能函数
    ax1.set_title('Potential')
    
    ax2 = plt.subplot(gs[1])
    ax2.plot((train_wavefunction[i]) ** 2, label='Wavefunction')  # 在第二个子图中绘制波函数
    ax2.set_title('Wavefunction')     
    ax1.legend(loc='upper right')      # 添加势能函数图例
    ax2.legend(loc='upper right')      # 添加波函数图例

    plt.tight_layout(h_pad=1.5)  # Adjust the vertical spacing between subplots
    plt.savefig(f"pic_of_wavefunction/sample_{i+1}.png")      
    plt.close()
#生成图片
for i in range(int(num_samples*train_ratio)):
    generate_image(i)
