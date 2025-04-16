import numpy as np
import pandas as pd
import sys, os
from sklearn.cluster import KMeans
os.environ["OMP_NUM_THREADS"] = "1"  # 禁用多线程
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

def extract_constellation_features(constellation_points):
    # 计算每个星座图点与理想点之间的平均偏差
    ideal_symbols = np.array([0.707+0.707j, -0.707+0.707j, -0.707-0.707j, 0.707-0.707j]) / np.sqrt(2)
    deviation = np.abs(constellation_points - ideal_symbols[:, np.newaxis])
    mean_deviation = np.mean(deviation, axis=1)

    # 计算星座图形状特征
    area = np.sum(deviation, axis=1)  # 面积特征
    perimeter = np.sum(deviation, axis=1)  # 周长特征

    # 返回特征向量
    features = np.hstack((mean_deviation, area, perimeter))
    return features

# 读取文件并分组
def read_and_split_iq_data(file_path, n_samples_per_group, n_groups, snr=None, flag='train'):
    """read iq_data and split it into n groups with n samples
    Args:
        file_path: 文件路径
        n_samples_per_group: 多少个样本点一组
        n_groups : 多少组
    Returns:
        data: 二维矩阵，每一行为一组
    """
    # 如果要生成测试数据，需要将组数改为1000，且从start开始采集
    if flag == "test":
        start = 25000000
        # n_groups = 1000
    else:
        start = 10000
    # 读取整个文件的数据，并进行归一化
    data = np.fromfile(file_path, dtype=np.complex64)
    data -= np.mean(data)
    data /= np.max(np.abs(data))
    if snr != None:
        sigma = np.sqrt(0.5 / (10**(snr / 10)))  # 计算标准差
        data += sigma * (np.random.randn(len(data)) + 1j * np.random.randn(len(data)))
    # 确保文件大小足以分成n_groups个组
    total_samples = n_samples_per_group * n_groups
    # if data.size < total_samples:
    #     raise ValueError("文件中的样本总数小于需要的样本数")
    # 重塑为 (n_groups, n_samples_per_group)
    data = data[start:total_samples+start].reshape(n_groups, n_samples_per_group)
    print(data.shape)
    return data

# 计算特征
def calculate_features(samples):
    features = {}
    kmeans = KMeans(n_clusters=4, random_state=0).fit(np.column_stack([samples.real, samples.imag]))
    cluster_centers = kmeans.cluster_centers_
        # 加入每个聚类中心的实部和虚部坐标作为特征
    for i, center in enumerate(cluster_centers):
        features[f'center_{i}_real'] = center[0]
        features[f'center_{i}_imag'] = center[1]
    # features['cluster_center_mean_i'] = np.mean(cluster_centers[:, 0])
    # features['cluster_center_mean_q'] = np.mean(cluster_centers[:, 1])
    features['mean_i'] = np.mean(samples.real)
    features['mean_q'] = np.mean(samples.imag)
    features['std_dev'] = np.std(np.abs(samples))
    features['phase_consistency'] = np.std(np.angle(samples))
    unique, counts = np.unique(samples, return_counts=True)
    probabilities = counts / counts.sum()
    features['entropy'] = -np.sum(probabilities * np.log2(probabilities))
    features['skewness'] = np.mean((samples.real - features['mean_i'])**3) / (np.std(samples.real)**3)
    features['kurtosis'] = np.mean((samples.real - features['mean_i'])**4) / (np.std(samples.real)**4)
    center_distance = np.abs(samples - (features['mean_i'] + 1j * features['mean_q']))
    features['center_distance_std'] = np.std(center_distance)
    labels = kmeans.labels_
    cluster_counts = np.array([np.sum(labels == i) for i in range(4)])
    features['cluster_counts_std'] = np.std(cluster_counts)
    features['label'] = 0  # 增加标签列
    # print(features)
    return features

# 主程序
# def process_iq_data(device, n_samples_per_group, n_groups, snr):
#     """分组加计算特征并存储csv
#     Args:
#         device: 处理设备类别
#         n_samples_per_group: 多少个样本点一组
#         n_groups : 多少组
#         snr: 信噪比
#     """
#     file_path = '../raw data/' + device + "/" + device + ".iq"
#     groups = read_and_split_iq_data(file_path, n_samples_per_group, n_groups)
#     feature_list = [calculate_features(group) for group in groups]
#     df = pd.DataFrame(feature_list)
#     output_file_name = '../preprocessed/' + device + "/" + device + "_" + str(n_samples_per_group) + ".csv"
#     df.to_csv(output_file_name, index=False)

def process_iq_data(device, n_samples_per_group, n_groups, snr_list, flag = "train"):
    """分组加计算特征并存储csv
    Args:
        device: 处理设备类别
        n_samples_per_group: 多少个样本点一组
        n_groups : 多少组
        snr: 信噪比
    """
    file_path = '../dataset/raw data/' + device + "/" + device + ".iq"
    # 首先处理不同样本数量，此时不加高斯白噪声
    for sample in n_samples_per_group:
        groups = read_and_split_iq_data(file_path, sample, n_groups, snr=None, flag=flag)
        feature_list = [calculate_features(group) for group in groups]
        df = pd.DataFrame(feature_list)
        # print(df.head())
        output_file_name = '../preprocessed/' + device + "/" + device + "_" + str(sample)
        if flag == "test":
            output_file_name += "_test.csv"
        else:
            output_file_name += ".csv"
        df.to_csv(output_file_name, index=False)
    # 再处理信噪比，信噪比默认样本数量为256
    for snr in snr_list:
        groups = read_and_split_iq_data(file_path, 256, n_groups, snr, flag=flag)
        feature_list = [calculate_features(group) for group in groups]
        df = pd.DataFrame(feature_list)
        output_file_name = '../preprocessed/' + device + "/" + device + "_" + str(snr)
        if flag == "test":
            output_file_name += "_test.csv"
        else:
            output_file_name += ".csv"
        df.to_csv(output_file_name, index=False)



def csv2bin(device):
    # 读取 CSV 文件  '.\\raw data\\limesdr/limesdr.csv'
    filename = '.\\raw data\\' + device + "/" + device + ".csv"
    data = pd.read_csv(filename, header=None)
    iq_data = data[0].values
    # 按照交替存储的顺序分割成实部和虚部
    real_data = iq_data[::2]
    imaginary_data = iq_data[1::2]
    # 将实部和虚部数据合并为复数
    iq_data = real_data + 1j * imaginary_data
    # 将复数数据转换为二进制格式
    binary_data = np.array(iq_data, dtype=np.complex64).tobytes()
    # 将二进制数据写入文件
    output_name = '.\\dataset\\raw data\\' + device + "/" + device + ".iq"
    with open(output_name, 'wb') as f:
        f.write(binary_data)
    print("IQ文件已创建")

# print(read_and_split_iq_data('./raw data/bladerf/bladerf.iq', 128, 2))

# 合并文件并分配新标签
def merge_and_label_csv(file_paths, labels):
    # 初始化空的DataFrame
    combined_df = pd.DataFrame()
    # 遍历每个文件及其标签
    for file_path, label in zip(file_paths, labels):
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # 更新标签
        df['label'] = label
        # 将更新后的DataFrame添加到总的DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

# 将所有不同类型的数据分别整合
def combine_all_snr(device_name, snr_list, labels):
    device_path_snr = [f'../preprocessed/{name}/{name}_{snr}.csv' for name in device_name for snr in snr_list]
    device_path_snr_test = [f'../preprocessed/{name}/{name}_{snr}_test.csv' for name in device_name for snr in snr_list]
    snr = [[device_path_snr[i] for i in range(j, len(device_path_snr), 13)] for j in range(13)]
    snr_t = [[device_path_snr_test[i] for i in range(j, len(device_path_snr_test), 13)] for j in range(13)]
    for i in range(len(snr)):
        print(snr[i])
        combined_snr = merge_and_label_csv(snr[i], labels)
        output_path_snr = '../preprocessed/combined_features_' + str(snr_list[i]) + '.csv'
        combined_snr.to_csv(output_path_snr, index=False)
        print(snr_t[i])
        combined_snr_t = merge_and_label_csv(snr_t[i], labels)
        output_path_snr_t = '../preprocessed/combined_features_' + str(snr_list[i]) + '_test.csv'
        combined_snr_t.to_csv(output_path_snr_t, index=False)

def combine_all_sym(device_name, sym_list, labels):
    device_path_sym = [f'../preprocessed/{name}/{name}_{sym}.csv' for name in device_name for sym in sym_list]
    device_path_sym_test = [f'../preprocessed/{name}/{name}_{sym}_test.csv' for name in device_name for sym in sym_list]
    sym = [[device_path_sym[i] for i in range(j, len(device_path_sym), 9)] for j in range(9)]
    sym_t = [[device_path_sym_test[i] for i in range(j, len(device_path_sym_test), 9)] for j in range(9)]
    for i in range(len(sym)):
        print(sym[i])
        combined_sym = merge_and_label_csv(sym[i], labels)
        output_path_sym = '../preprocessed/combined_features_' + str(sym_list[i]) + '.csv'
        combined_sym.to_csv(output_path_sym, index=False)
        print(sym_t[i])
        combined_sym_t = merge_and_label_csv(sym_t[i], labels)
        output_path_sym_t = '../preprocessed/combined_features_' + str(sym_list[i]) + '_test.csv'
        combined_sym_t.to_csv(output_path_sym_t, index=False)
