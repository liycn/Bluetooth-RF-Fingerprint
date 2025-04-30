import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import rcParams

def generate_bluetooth_iq(output_file="bluetooth_signal.iq", duration_sec=0.5, sample_rate=20e6):
    """
    生成模拟蓝牙设备的IQ数据并保存为二进制.iq文件
    
    参数:
        output_file: 输出的.iq文件名
        duration_sec: 信号持续时间(秒)
        sample_rate: 采样率(Hz)
    """
    # 蓝牙相关参数
    bt_symbol_rate = 1e6  # 蓝牙的符号速率(1MHz)
    freq_deviation = 250e3  # 频偏(蓝牙GFSK调制的频偏约为250kHz)
    bt_h_index = 0.5  # 调制指数
    channel_spacing = 2e6  # 蓝牙通道间隔(2MHz)
    
    # 生成随机的二进制数据(模拟蓝牙数据包)
    num_symbols = int(duration_sec * bt_symbol_rate)
    binary_data = np.random.randint(0, 2, num_symbols)
    
    # 应用高斯滤波器进行脉冲整形(实现GFSK调制)
    # 高斯滤波器的BT积为0.5(蓝牙标准)
    bt_product = 0.5
    span = 4
    sps = int(sample_rate / bt_symbol_rate)  # 每符号采样点数
    
    # 上采样
    upsampled_data = np.zeros(num_symbols * sps)
    upsampled_data[::sps] = 2 * binary_data - 1  # NRZ编码(0->-1, 1->1)
    
    # 设计高斯滤波器 - 修正版，不使用signal.gaussian
    filter_length = span * sps
    time_idx = np.arange(-(filter_length//2), filter_length//2 + filter_length%2)
    std = sps / (2 * np.pi * bt_product)
    gaussian_filter = np.exp(-(time_idx**2) / (2 * std**2))
    gaussian_filter = gaussian_filter / np.sum(gaussian_filter)
    
    # 应用高斯滤波器
    filtered_data = np.convolve(upsampled_data, gaussian_filter, mode='same')
    
    # 对滤波后的信号进行积分得到相位
    phase = np.cumsum(filtered_data) * freq_deviation * (2 * np.pi / sample_rate)
    
    # 生成IQ信号
    i_samples = np.cos(phase)
    q_samples = np.sin(phase)
    
    # 添加一些噪声来模拟实际蓝牙信号
    noise_power = 0.01
    i_samples += np.sqrt(noise_power/2) * np.random.randn(len(i_samples))
    q_samples += np.sqrt(noise_power/2) * np.random.randn(len(q_samples))
    
    # 为了模拟真实场景，加入一些上升下降沿和频率漂移
    # 频率漂移
    t = np.arange(len(i_samples)) / sample_rate
    drift = np.sin(2 * np.pi * 0.2 * t) * 0.05  # 慢频率漂移
    phase_drift = np.cumsum(drift)
    
    i_samples_drift = i_samples * np.cos(phase_drift) - q_samples * np.sin(phase_drift)
    q_samples_drift = i_samples * np.sin(phase_drift) + q_samples * np.cos(phase_drift)
    
    # 应用幅度包络(模拟信号开始和结束)
    ramp_length = int(0.05 * len(i_samples))
    window = np.ones(len(i_samples))
    window[:ramp_length] = np.linspace(0, 1, ramp_length)
    window[-ramp_length:] = np.linspace(1, 0, ramp_length)
    
    i_samples_final = i_samples_drift * window
    q_samples_final = q_samples_drift * window
    
    # 将IQ数据交织并保存为二进制文件
    with open(output_file, 'wb') as f:
        for i, q in zip(i_samples_final, q_samples_final):
            # 使用32位浮点格式保存IQ样本
            f.write(struct.pack('ff', float(i), float(q)))
    
    print(f"已生成蓝牙IQ文件: {output_file}")
    print(f"文件信息:")
    print(f" - 采样率: {sample_rate/1e6} MHz")
    print(f" - 持续时间: {duration_sec} 秒")
    print(f" - 样本数: {len(i_samples_final)}")
    print(f" - 文件大小: {8*len(i_samples_final)/1024/1024:.2f} MB (32位浮点IQ数据)")
    
    return i_samples_final, q_samples_final

def plot_iq_data(i_data, q_data, sample_rate, title="蓝牙IQ信号"):
    """绘制IQ数据的图表"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体，支持中文
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示异常
    # 时域图
    plt.figure(figsize=(15, 10))
    
    # I和Q波形
    plt.subplot(2, 2, 1)
    t = np.arange(len(i_data)) / sample_rate * 1000  # 转换为毫秒
    max_t = 0.1  # 仅显示前0.1毫秒以便于观察
    show_samples = int(max_t * sample_rate / 1000)
    plt.plot(t[:show_samples], i_data[:show_samples], 'b-', label='I')
    plt.plot(t[:show_samples], q_data[:show_samples], 'r-', label='Q')
    plt.xlabel('时间 (毫秒)')
    plt.ylabel('幅度')
    plt.title('IQ时域波形 (局部)')
    plt.legend()
    plt.grid(True)
    
    # IQ轨迹图
    plt.subplot(2, 2, 2)
    plt.plot(i_data, q_data, 'g.', alpha=0.1)
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.title('IQ轨迹图')
    plt.grid(True)
    plt.axis('equal')
    
    # 功率谱密度
    plt.subplot(2, 2, 3)
    complex_signal = i_data + 1j * q_data
    f, psd = signal.welch(complex_signal, sample_rate, nperseg=1024)
    plt.semilogy(f/1e6, psd)
    plt.xlabel('频率 (MHz)')
    plt.ylabel('功率谱密度 (dB/Hz)')
    plt.title('功率谱密度')
    plt.grid(True)
    
    # 频谱图
    plt.subplot(2, 2, 4)
    f, t, Sxx = signal.spectrogram(complex_signal, sample_rate, nperseg=256, noverlap=128)
    plt.pcolormesh(t, f/1e6, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('频率 (MHz)')
    plt.xlabel('时间 (秒)')
    plt.title('频谱图')
    plt.colorbar(label='功率 (dB)')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('bluetooth_iq_analysis.png')
    plt.show()