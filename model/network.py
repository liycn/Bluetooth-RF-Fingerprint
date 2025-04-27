import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

# 复数张量的表示：原通道数为n，则将通道数分为2n，前n表示实部，后n表示虚部。

# 模型参数配置 
IMAGE_SIZE = 28
IMAGE_CHANNEL = 2  # 实部+虚部
FILTER1_SIZE = 3
FILTER1_NUM = 64
FILTER2_SIZE = 3 
FILTER2_NUM = 128
FILTER3_SIZE = 3
FILTER3_NUM = 256
FILTER4_SIZE = 3
FILTER4_NUM = 512
FC1_SIZE = 256
OUTPUT_NODE = 4  # 分类数量（默认为3个类别：bladerf, hackrf0, hackrf1, limesdr）

class ComplexConv2d(nn.Module):
    """复数卷积层实现"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv2d, self).__init__()
        # 实部到实部的卷积
        self.conv_rr = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # 实部到虚部的卷积
        self.conv_ri = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # 虚部到实部的卷积
        self.conv_ir = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # 虚部到虚部的卷积
        self.conv_ii = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        # 输入x形状为[batch, 2*in_channels, height, width]
        # 前一半通道是实部，后一半通道是虚部
        
        # 分离实部和虚部
        real_part = x[:, :x.size(1)//2]
        imag_part = x[:, x.size(1)//2:]
        
        # 复数卷积运算
        # (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
        real_result = self.conv_rr(real_part) - self.conv_ii(imag_part)
        imag_result = self.conv_ri(real_part) + self.conv_ir(imag_part)
        
        # 合并实部和虚部，返回形状为[batch, 2*out_channels, height, width]
        return torch.cat([real_result, imag_result], dim=1)

class ModReLU(nn.Module):
    """ModReLU激活函数"""
    def __init__(self, channels):
        super(ModReLU, self).__init__()
        self.bias = nn.Parameter(torch.zeros(channels))
    
    def forward(self, x):
        # 输入x形状为[batch, 2*channels, height, width]
        # 前一半通道是实部，后一半通道是虚部
        
        # 分离实部和虚部
        real = x[:, :x.size(1)//2]
        imag = x[:, x.size(1)//2:]
        
        # 计算模值
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-5)
        
        # 应用偏置
        bias_view = self.bias.view(1, -1, 1, 1)
        magnitude_out = magnitude - bias_view
        
        # 计算激活系数
        scale = F.relu(magnitude_out) / (magnitude + 1e-5)
        
        # 应用激活
        real_out = real * scale
        imag_out = imag * scale
        
        # 合并实部和虚部
        return torch.cat([real_out, imag_out], dim=1)

class ResidualBlock(nn.Module):
    """残差块实现"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        
        # 每个通道分为实部和虚部，所以通道数要乘以2
        in_ch = in_channels * 2
        out_ch = out_channels * 2
        
        # 主路径
        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.modrelu1 = ModReLU(out_channels)
        self.dropout = nn.Dropout2d(0.2)
        
        # 残差连接
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = ComplexConv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        # 主路径
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.modrelu1(out)
        out = self.dropout(out)
        
        # 残差连接
        if self.shortcut:
            identity = self.shortcut(identity)
        
        out = out + identity
        return out

class ComplexCNN(nn.Module):
    """基于PyTorch网络结构的复数CNN实现"""
    def __init__(self, num_features, num_classes=3):
        super(ComplexCNN, self).__init__()
        
        # 计算输入特征的形状
        side_length = int(np.ceil(np.sqrt(num_features)))
        
        # 第一层卷积
        self.conv1 = ComplexConv2d(1, FILTER1_NUM//2, FILTER1_SIZE, padding=1)
        self.bn1 = nn.BatchNorm2d(FILTER1_NUM)
        self.modrelu1 = ModReLU(FILTER1_NUM//2)
        self.dropout1 = nn.Dropout2d(0.2)
        
        # 残差块
        self.res_block1 = ResidualBlock(FILTER1_NUM//2, FILTER2_NUM//2)
        self.res_block2 = ResidualBlock(FILTER2_NUM//2, FILTER3_NUM//2)
        self.res_block3 = ResidualBlock(FILTER3_NUM//2, FILTER4_NUM//2)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 修正：确保全连接层的输入输出维度匹配
        # FILTER4_NUM//2是因为我们只使用了特征的绝对值，没有分开实部和虚部
        self.fc1 = nn.Linear(FILTER4_NUM//2, FC1_SIZE)
        self.fc2 = nn.Linear(FC1_SIZE, num_classes)
    
    def forward(self, x):
        # 输入x形状为[batch, 2, height, width]
        # 调整通道顺序，使前一半通道为实部，后一半通道为虚部
        real = x[:, 0:1]  # 实部
        imag = x[:, 1:2]  # 虚部
        x = torch.cat([real, imag], dim=1)  # [batch, 2, height, width]
        
        # 第一层卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.modrelu1(x)
        x = self.dropout1(x)
        
        # 残差块
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # 全局池化
        x = self.global_pool(x)
        
        # 准备全连接层输入
        real = x[:, :x.size(1)//2].view(x.size(0), -1)
        imag = x[:, x.size(1)//2:].view(x.size(0), -1)
        
        # 复数绝对值作为特征
        x = torch.sqrt(real**2 + imag**2 + 1e-5)
        
        # 修正：确保输入维度与权重匹配
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# 读取IQ数据文件
def load_iq_file(file_path):
    """读取原始IQ数据文件"""
    # 根据你的.iq文件格式调整读取方式
    # 通常.iq文件存储复数数据，I和Q交替存储
    try:
        # 尝试以复数形式读取
        data = np.fromfile(file_path, dtype=np.complex64)
    except:
        # 如果失败，尝试以浮点数读取然后转换为复数
        data = np.fromfile(file_path, dtype=np.float32)
        # 假设数据是I和Q交替的形式
        data = data[::2] + 1j * data[1::2]
    
    return data


class ComplexSignalDataset(Dataset):
    """复数信号数据集"""
    def __init__(self, features, labels, scaler_real=None, scaler_imag=None):
        # 分离实部和虚部
        real_part = np.real(features)
        imag_part = np.imag(features)
        
        # 标准化
        if scaler_real is not None and scaler_imag is not None:
            real_part = scaler_real.transform(real_part)
            imag_part = scaler_imag.transform(imag_part)
        
        # 调整数据维度为正方形
        n_features = real_part.shape[1]
        side_length = int(np.ceil(np.sqrt(n_features)))
        padding = side_length * side_length - n_features
        
        # 填充零
        real_part = np.pad(real_part, ((0, 0), (0, padding)), 'constant')
        imag_part = np.pad(imag_part, ((0, 0), (0, padding)), 'constant')
        
        # 重塑为正方形
        real_part = real_part.reshape(-1, side_length, side_length)
        imag_part = imag_part.reshape(-1, side_length, side_length)
        
        # 合并实部和虚部通道
        self.data = torch.FloatTensor(np.stack([real_part, imag_part], axis=1))
        self.labels = torch.LongTensor(labels.values if hasattr(labels, 'values') else labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_model(num_features, num_classes=3):
    """创建复数CNN模型"""
    return ComplexCNN(num_features, num_classes)

def create_dataloaders(features_train, labels_train, features_test, labels_test, batch_size=32):
    """创建训练和测试数据加载器"""
    # 创建标准化器
    scaler_real = StandardScaler().fit(np.real(features_train))
    scaler_imag = StandardScaler().fit(np.imag(features_train))
    
    # 创建数据集
    train_dataset = ComplexSignalDataset(features_train, labels_train, scaler_real, scaler_imag)
    test_dataset = ComplexSignalDataset(features_test, labels_test, scaler_real, scaler_imag)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, (scaler_real, scaler_imag) 

def train_and_evaluate(model, train_loader, test_loader, device, num_epochs=30):
    """训练模型并记录训练过程中的准确率"""
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 记录训练和测试准确率
    train_accuracies = []
    test_accuracies = []
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f'第 {epoch+1}/{num_epochs} 轮,')
        # 训练
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        train_accuracies.append(train_acc)
        print(f'♂️ trainLoss: {train_loss/len(train_loader):.4f}, trainAcc: {train_acc:.2f}%')
        
        # 验证
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        test_accuracies.append(test_acc)
        print(f'♀️ testLoss: {test_loss/len(test_loader):.4f}, testAcc: {test_acc:.2f}%')
    
    return train_accuracies, test_accuracies

def get_predictions(model, data_loader, device):
    """获取模型在给定数据上的预测结果"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="获取预测结果"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, class_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()
    
    # 计算每个类别的精确率和召回率
    print("精确率和召回率:")
    for i, class_name in enumerate(class_names):
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() != 0 else 0
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() != 0 else 0
        print(f"{class_name} - 精确率: {precision:.4f}, 召回率: {recall:.4f}")

def plot_accuracy_curve(train_accuracies, test_accuracies):
    """绘制训练和测试准确率的折线图"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_accuracies) + 1)
    plt.plot(epochs, train_accuracies, 'b-', label='训练准确率')
    plt.plot(epochs, test_accuracies, 'r-', label='测试准确率')
    plt.title('训练和测试准确率随轮次的变化')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_curve.png', dpi=300)
    plt.show()