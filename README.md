<img src="./README/CQU-EIE.svg">
<h1 align="center">车载蓝牙设备的射频指纹提取技术设计与实现</h1>

## 🎈 项目

**基础环境**：`Windows 11 专业工作站版 24H2` 、 `Anaconda3 2020.11(Python 3.8.5 64-bit)` 

1. 进入项目模型的目录

    ```bash
    cd .\model\
    ```

2. 然后创建项目的 `conda` 环境

    ```bash
    conda env create -n bt python=3.10.16
    ```

3. 激活环境

    ```bash
    conda activate bt
    ```

4. 更换 `pip` 源

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```

5. 下载项目所有包

    ```bash
    pip install -r requirements.txt
    ```

<!-- ## 项目结构流程

|序号|流程|对应文件|
|:-|:-|:-|
|1|数据采集|这个部分就用 GNU Radio + HackRF One实现的|
|2|特征提取|[`model/utility.py`](./model/utility.py) 核心处理函数<br> [`model/getFeature.ipynb`](./model/getFeature.ipynb) 特征提取<br> |
|3|特征选择|[`model/feature_selection.ipynb`](./model/feature_selection.ipynb) 选择最优特征|
|4|模型训练|**[`model/network.py`](./model/network.py) 复数卷积神经网络的网络结构**<br> [`model/complexCNN_snr.ipynb`](./model/complexCNN_snr.ipynb) 基于信噪比的复数CNN模型训练<br> [`model/complexCNN_sym.ipynb`](./model/complexCNN_sym.ipynb) 基于号长度的复数CNN模型训练<br> [`model/xgboost_snr.ipynb`](./model/xgboost_snr.ipynb) 基于信噪比的XGBoost模型训练<br> [`model/xgboost_sym.ipynb`](./model/xgboost_sym.ipynb) 基于符号长度的XGBoost模型训练<br> [`model/xgboost_alg_snr.ipynb`](./model/xgboost_alg_snr.ipynb) 改进的XGBoost模型训练（信噪比）<br> [`model/xgboost_alg_sym.ipynb`](./model/xgboost_alg_sym.ipynb) 改进的XGBoost模型训练（符号长度）<br> |
|5|模型评估|| -->

## 项目摘要

随着新能源汽车的快速发展，车载系统的安全也越发重要。蓝牙模块作为其中的一环，其开放式通信的方式在身份认证和防伪技术方面仍然会造成安全性的难题。蓝牙设备的非法设备接入车辆可能导致安全漏洞，而黑客通过劫持蓝牙连接实施攻击、窃取隐私数据或干扰正常通信，进一步加剧了车辆安全隐患。为此，本文提出了一种基于射频指纹的蓝牙设备识别技术，通过无线通信设备的独特的射频指纹对设备进行准确识别，极大地增强了车载系统蓝牙模块的安全性。
研究通过低功耗蓝牙信号中的I/Q数据作为射频指纹对象进行采集后，通过提取I/Q数据中的符号长度和信噪比两个方面进行特征选择，采用包含XGBoost、SVM、KNN等传统机器学习形成的集成学习与复数卷积神经网络的深度学习两种方式进行对照实验。