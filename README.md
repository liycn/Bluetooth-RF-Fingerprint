<img src="./README/CQU-EIE.svg">
<h1 align="center">车载蓝牙设备的射频指纹提取技术设计与实现</h1>

---

## 项目概述

本项目是我在重大卓工的**本科毕业设计**。

项目聚焦于低功耗蓝牙（BLE）设备的射频指纹提取技术，旨在通过研究和开发射频指纹提取技术来识别并保证车载蓝牙设备的安全性和身份识别。

## 研究方向

**低功耗蓝牙** | **通信** | **射频指纹** | **模型训练测试**

## 主要内容

1. **探索和分析低功耗蓝牙（BLE）设备的射频特征**
2. **研究射频指纹提取的理论基础和技术实现方法**
3. **设计并实现一种基于深度学习或信号处理算法的指纹提取模型**
4. **利用实验数据进行模型训练和测试**
5. **完成系统实现并对系统性能进行评估和改进**


## 准备工作

- 硬件：
  - **HackRF One** 用于捕获 BLE 信号
  - **天线** 确保使用支持 2.4 GHz 频段的天线

- 软件：
  - **GNU Radio 3.7.0** 用于信号处理和调制解调，`sudo apt install gnuradio`。
  - **gr-bluetooth** GNU Radio 的蓝牙模块，用于解调 BLE 信号，在 github 上 clone 就好了 地址 [https://github.com/greatscottgadgets/gr-bluetooth.git](https://github.com/greatscottgadgets/gr-bluetooth.git)
    - 前提：git、cmake、
    - 步骤
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
  - **python** 用于后续的信号处理和特征提取。
