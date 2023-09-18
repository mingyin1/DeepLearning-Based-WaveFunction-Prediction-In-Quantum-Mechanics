import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time
import argparse
import os
import shutil
import concurrent.futures


parser = argparse.ArgumentParser(description='Matrix Factorization')
parser.add_argument('--device', choices=['cpu', 'gpu'])
parser.add_argument('--lr', type = float, default = 1e-5)
parser.add_argument('--round', type = int , default = 50000)

args = parser.parse_args()
device = args.device
if device == 'gpu':
    device = torch.device("cuda")  # 使用GPU设备
    print("GPU is available")
else:
    device = torch.device("cpu")  # 使用CPU设备
    print("Using CPU")
print(f'learning rate:{args.lr}')
print(f'num_of_epoch:{args.round}')
# 定义神经网络模型
class WavefunctionPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(WavefunctionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        return x

# 读取训练集和测试集的势能和波函数数据
train_potential = np.load("data/train_potential.npy")
train_wavefunction = np.load("data/train_wavefunction.npy")
test_potential = np.load("data/test_potential.npy")
test_wavefunction = np.load("data/test_wavefunction.npy")

# 检查波函数数据的维度并调整
if train_wavefunction.ndim == 3:
    train_wavefunction = train_wavefunction.squeeze()
if test_wavefunction.ndim == 3:
    test_wavefunction = test_wavefunction.squeeze()

# 转换数据为PyTorch的Tensor类型并移动到GPU
train_potential = torch.Tensor(train_potential).to(device)
train_wavefunction = torch.Tensor(train_wavefunction).to(device)
test_potential = torch.Tensor(test_potential).to(device)
test_wavefunction = torch.Tensor(test_wavefunction).to(device)

# 构建神经网络模型并移动到GPU
input_size = len(train_potential[0])
output_size = len(train_wavefunction[0])
model = WavefunctionPredictor(input_size, output_size).to(device)
lr = args.lr
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练神经网络模型
t_begin = time.time()
num_epochs = args.round

for epoch in range(num_epochs):
    # 前向传播
    outputs = model(train_potential)
    loss = criterion(outputs, train_wavefunction)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印损失值
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    #动态调整学习率
    if (epoch+1) == 10000 or (epoch+1) == 40000:
        lr = lr/10
        print(f'Ajust learning rate to:{lr}')
# 测试神经网络模型
model.eval()
with torch.no_grad():
    test_outputs = model(test_potential)
time_cost = time.time()-t_begin
print(f'time cost: {time_cost}')

torch.save(model, 'model.pth')  #保存模型

# 将数据移回到CPU并转换为NumPy数组
test_potential = test_potential.cpu().numpy()
test_wavefunction = test_wavefunction.cpu().numpy()
test_outputs = test_outputs.cpu().numpy()

# 可视化预测波函数和实际波函数
num_samples = len(test_potential)

if os.path.exists('pic_of_predict'):
    shutil.rmtree('pic_of_predict')
os.makedirs('pic_of_predict')
print(f"Directory pic_of_predict created!")

def generate_image(i):
    fig = plt.figure(figsize=(8, 6))                 
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5])   # 创建包含两个子图的画布
    
    ax1 = plt.subplot(gs[0])
    ax1.plot(test_potential[i], label='Potential')   # 在第一个子图中绘制势能函数
    ax1.set_title('Potential')
    
    ax2 = plt.subplot(gs[1])
    ax2.plot((test_wavefunction[i]) ** 2, label='Actual Wavefunction')  # 在第二个子图中绘制波函数
    ax2.plot((test_outputs[i]) ** 2, label = 'Predicted Wavefunction')
    ax2.set_title('Wavefunction')     
    ax1.legend(loc='upper right')      # 添加势能函数图例
    ax2.legend(loc='upper right')      # 添加波函数图例

    plt.tight_layout(h_pad=1.5)  # Adjust the vertical spacing between subplots
    plt.savefig(f"pic_of_predict/sample_{i+1}.png")      
    plt.close()

for i in range(num_samples):
    generate_image(i)
