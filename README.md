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

## 项目结构流程

|序号|流程|对应文件|
|:-|:-|:-|
|1|数据采集||
|2|特征提取|`model/utility.py` 核心处理函数<br> `model/getFeature.ipynb` 特征提取<br> |
|3|特征选择|`model/feature_selection.ipynb` 选择最优特征|
|4|模型训练|**`model/network.py` 复数卷积神经网络的网络结构**<br> `model/complexCNN_snr.ipynb` 基于信噪比的复数CNN模型训练<br> `model/complexCNN_sym.ipynb` 基于号长度的复数CNN模型训练<br> `model/xgboost_snr.ipynb` 基于信噪比的XGBoost模型训练<br> `model/xgboost_sym.ipynb` 基于符号长度的XGBoost模型训练<br> `model/xgboost_alg_snr.ipynb` 改进的XGBoost模型训练（信噪比）<br> `model/xgboost_alg_sym.ipynb` 改进的XGBoost模型训练（符号长度）<br> |
|5|模型评估||

### 可视化分析

- `constellation.ipynb`：分析信号的星座图

- `test.py`：分析不同因素对星座图的影响