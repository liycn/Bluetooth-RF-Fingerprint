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

## 项目背景

随着物联网技术的突飞猛进，智能设备间通过无线通信互联互动，构建了庞大的物联网生态，同时在新能源汽车快速发展的背景下，车载系统作为两者之间的桥梁，连接了物联网设备与新能源汽车，进一步凸显了车载系统在两者间的核心地位。在车载系统的桥梁中，车载蓝牙作为关键的组成部分，发挥着至关重要的作用。

车载蓝牙设备作为车载系统中信息娱乐、定位导航、设备互联等应用的关键接口，极大地提升了车载系统的功能性和用户体验。但是蓝牙作为开放式的通信方式，也伴随着潜在的安全风险，特别是在设备身份认证和防伪技术等方面。例如通过蓝牙设备进行身份伪造可以实现非法设备介入车辆，从而威胁车辆的安全。此外黑客还可能在蓝牙通信阶段不断实施攻击，窃取车主隐私数据或者干扰车辆正常通信，进一步加剧了车载系统甚至车辆本身的安全隐患[1]。

传统的解决方案主要集中在密码、密钥或认证码等方向，在复杂的物联网环境中存在严重的局限性，极易受到暴力破解或泄密攻击，依旧不能有效地解决安全性问题。其根本原因是传统的解决方案是基于设备软件层面的安全协议或安全机制，比如身份验证配对、密钥交换、安全连接等，同时攻击者就是通过找到安全协议或具体实现中的例如密码猜测、协议缺陷利用、缓冲区溢出等漏洞进行攻击，所以即便安全方案十分优秀，都仍有会被攻破的可能。

针对传统解决方案中基于软件层面的局限性，射频指纹识别作为一种基于物理层面特征的安全性方案，提供了更为可靠的身份验证方式。射频指纹利用设备发射无线电信号的独特物理特征进行身份认证，如同生物指纹特征般具有唯一性，能够有效识别设备真实身份。其描述的是一台设备“是谁”，而是不是“凭证是什么”，也就是告诉配对者这是哪台设备，并不关心这台设备在软件层面受到怎样的安全保护机制。并且射频指纹不需要直接接触设备或采集蓝牙传输中的敏感数据，具有更优秀的隐私保护性。

射频指纹识别不仅能克服传统识别方案的不足，还兼具低成本、低功耗的优势，尤其适合车载系统中的低功耗蓝牙（BLE）设备。传统的部分认证方案还需要额外的计算资源与硬件支持，但是射频指纹可以直接利用无线通信设备本身所发射的无线电信号进行识别，不需要额外的硬件或者过多的计算资源，为蓝牙设备的认证成本降低了部分成本[2]。

综上所述，射频指纹技术为蓝牙提供了一种更加安全、更具隐私性、功耗与成本更低的身份认证方案，尤其适用于资源受限的车载系统，通过有效防止未授权设备的接入来提高车载系统的安全性问题。随着物联网技术的日益发展，未来射频指纹技术不仅能在车载系统中发挥重要作用，更对物联网安全认证技术领域的发展奠定着重要的理论基石与实践意义。