from gnuradio import gr
import numpy as np

class preamble_extractor(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name='Preamble Extractor',
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.preamble_pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        self.samples_per_symbol = 4
        self.preamble_length = len(self.preamble_pattern) * self.samples_per_symbol
        self.threshold = 0.8  # 相关性检测阈值
        
    def work(self, input_items, output_items):
        in_data = input_items[0]
        out_data = output_items[0]
        
        # 使用滑动窗口检测前导码
        for i in range(len(in_data) - self.preamble_length):
            window = in_data[i:i + self.preamble_length]
            # 进行前导码检测
            if self.detect_preamble(window):
                print("检测到前导码！")
                # 输出前导码部分
                out_data[i:i + self.preamble_length] = window
                
        return len(output_items[0])
        
    def detect_preamble(self, samples):
        # 将复数样本转换为实数
        real_samples = np.real(samples)
        
        # 对样本进行归一化
        real_samples = real_samples / np.max(np.abs(real_samples))
        
        # 使用相关性检测
        correlation = np.correlate(real_samples, self.preamble_pattern, mode='valid')
        
        # 检查相关性是否超过阈值
        return np.max(correlation) > self.threshold 