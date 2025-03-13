<img src="./README/CQU-EIE.svg">
<h1 align="center">车载蓝牙设备的射频指纹提取技术设计与实现</h1>

---

## 🎈 项目概述

本项目是我在重大卓工的**本科毕业设计**。

项目主要是提取低功耗蓝牙（BLE）设备的射频指纹，旨在通过研究和开发射频指纹提取技术来识别并保证车载蓝牙设备的安全性和身份识别。

<!-- ## 研究方向 -->

<!-- **低功耗蓝牙** | **通信** | **射频指纹**-->

## 🔬 主要内容

1. 探索和分析低功耗蓝牙（BLE）设备的射频特征
2. 研究射频指纹提取的理论基础和技术实现方法
3. 设计并实现一种基于深度学习或信号处理算法的指纹提取模型
4. 利用实验数据进行模型训练和测试
5. 完成系统实现并对系统性能进行评估和改进

## 🌳 项目环境

|类别|名称-版本(固件)|详情|
|:-:|:-|:-|
|~~系统~~|~~`ubuntu` 18.04~~|废案|
|系统|`Windows` 11|专业版|
|~~编程语言~~|~~`Python` 3.8.5~~|废案|
|编程语言|`Python` 3.8.5|作为模型训练的基础语言|
|程序工具|`Jupyter Notebook` 6.1.4|用于模型训练的基础|
|数据采集 硬件部分|`HackRF One` 2018.01.1 (API:1.02)|SDR 设备皆可。|
|蓝牙信息采集工具|`BluetoothView` 1.70|主要用于收集指定设备的mac地址和信道|
|~~数据采集 软件部分~~|~~`GNU Radio` 3.7.11~~|废案|
|数据采集 软件部分|`GNU Radio` 3.10.10.0|软件 Python 版本 3.11.9|
|~~GNU Radio-插件~~|~~gr-osmosdr~~|~~`sudo apt-get install gnuradio gr-osmosdr hackrf`~~ 貌似新版本不用自己单独下载。|
|~~GNU Radio-插件~~|~~gr-bluetooth~~|~~[仓库地址](https://github.com/greatscottgadgets/gr-bluetooth)~~ 废案|

### 📡 HankRF One

Ubuntu 上测试 HackRF One 的有效性

```BASH
hackrf_transfer -r test.raw -f 2480000000 -s 10000000 -g 40 -l 20 -a 1
```

> - -r test.raw：保存原始数据到文件
> - -f 2480000000：中心频率 2.48 GHz
> - -s 10000000：采样率 10 MHz
> - -g 40：射频增益 40 dB
> - -l 20：中频增益 20 dB
> - -a 1：启用天线供电（如果需要）

 Windows还没找到，感觉挺麻烦的...因为我最开始是在Ubuntu上做的原始数据采集，后面才转到windows上，后面看看能不能实现上测试有效性。

 但至少在windows上面我没有下载驱动

***

## 🔧 准备工作

### 🕹️ 硬件：

- **HackRF One** 用于捕获 BLE 信号
- **天线** 确保使用支持 2.4 GHz 频段的天线

### 🖥️ 软件：

- **GNU Radio (~~3.7.0~~)3.10.10.0** 用于信号处理和调制解调，`sudo apt install gnuradio` ，windows直接去官网下载[链接](https://wiki.gnuradio.org/index.php/InstallingGR)。
- **（废案）** gr-bluetooth： GNU Radio 的蓝牙模块，用于解调 BLE 信号，在 github 上 clone 就好了 [地址](https://github.com/greatscottgadgets/gr-bluetooth.git)： 
  ```BASH
  git clone https://github.com/greatscottgadgets/gr-bluetooth.git
  cd gr-bluetooth
  mkdir build
  cd build
  cmake ..
  make
  sudo make install
  sudo ldconfig
  ```

***

## 😡 数据采集（恼火）

现在有一个严重的问题就是，因为目前还没有调通指定目标设备的蓝牙信号提取，所以提取的蓝牙信号很多，频率又是 10MHz，所以信息量巨大，5分钟的连续提取就是 21.6GB 的数据，数据长度高达 29亿，所以才从想通过指定目标设备的方式进行数据提取，But, HackRF One不能指定目标设备，所以还是要采集大量设备。

话又说回来了，射频指纹应该采集的是设备连续很长一段时间的数据，因为射频指别其实说白了和人脸是一样的，短时间内的射频指纹数据基本没什么变化，所以数据存储应该是目前来说遇到的最大的问题。

目前数据采集部分有点问题，数据量非常大，所以准备先看看网上的数据集进。

但我看了一下网上的数据集后发现，他是通过捕捉**每个设备在几个月**内发出的信号来构建的，但是数据又只有16GB，所以可以看出来我的**数据有效性非常非常低**，现在要想一个办法在GNU Radio中保证数据的有效性，即指定目标设备。

### 🎯 所以目前有这样的思路（还没有提取射频指纹，只是提取目标设备的蓝牙数据）

1. HackRF 在 GNU Radio 里捕获 BLE 广播信号（IQ 格式）。
2. 在 Python 里用 scapy 或 gr-bluetooth 解析 BLE 数据包。
3. 根据目标设备的 MAC 地址筛选出目标设备的数据包。

#### 🚀 Python 中解析 BLE 数据包思路

|步骤|目的|说明|
|:-|:-|:-|
|**IQ 信号读取**|获取 HackRF 采集的 IQ 数据|.dat 文件中的复数数据|
|**解调信号**|将 IQ 信号转换为比特流|通过 FM 解调和相位差提取数据|
|**降采样**|BLE 带宽为 2 MHz，HackRF 采样率为 4 MHz → 需要降采样|`scipy.signal.decimate`|
|**转换比特流**|将解调后的数据转换为字节流|用于 BLE 解码|
|**scapy 解析 BLE 包**|解析 BLE 广播包结构|用 BTLE() 解析广播包|
|**根据 MAC 地址筛选**|只提取目标设备的广播包|通过 pkt.addr 进行 MAC 地址匹配|

~~[代码详情](./BluetoothSignal/ScreeningEquipment.ipynb)~~ 有问题，服了...

由于数据采集部分还有问题，先用网上的数据集做个模型出来交差，[数据集地址](https://zenodo.org/records/3876140#.YBJRIvlKiHs)

<!-- - **python** 用于后续的信号处理和特征提取。 -->