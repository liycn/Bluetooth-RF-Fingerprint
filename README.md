<img src="./README/CQU-EIE.svg">
<h1 align="center">车载蓝牙设备的射频指纹提取技术设计与实现</h1>

---

## 项目概述

本项目是我在重大卓工的**本科毕业设计**。

项目聚焦于低功耗蓝牙（BLE）设备的射频指纹提取技术，旨在通过研究和开发射频指纹提取技术来识别并保证车载蓝牙设备的安全性和身份识别。

<!-- ## 研究方向 -->

<!-- **低功耗蓝牙** | **通信** | **射频指纹**-->

## 主要内容

1. **探索和分析低功耗蓝牙（BLE）设备的射频特征**
2. **研究射频指纹提取的理论基础和技术实现方法**
3. **设计并实现一种基于深度学习或信号处理算法的指纹提取模型**
4. **利用实验数据进行模型训练和测试**
5. **完成系统实现并对系统性能进行评估和改进**


## 项目环境

|类别|名称-版本(固件)|详情|
|:-|:-|:-|
|系统|Ubuntu-18.04|非虚拟机，虚拟机也是可以的，只不过我没试过|
|编程语言|python-2.7.17|无|
|数据采集部分-SDR 硬件|HackRF One-2018.01.1 (API:1.02)||
|数据采集部分-软件|GNU Radio-3.7.11||
|GNU Radio-插件|gr-bluetooth|[github 地址](https://github.com/greatscottgadgets/gr-bluetooth#) clone下来编译就好了|
|GNU Radio-插件|gr-osmosdr|`sudo apt-get install gnuradio gr-osmosdr hackrf`|


### HankRF One相关

测试 HackRF One 的有效性

```BASH
hackrf_transfer -r test.raw -f 2480000000 -s 10000000 -g 40 -l 20 -a 1
```

> - -r test.raw：保存原始数据到文件
> - -f 2480000000：中心频率 2.48 GHz
> - -s 10000000：采样率 10 MHz
> - -g 40：射频增益 40 dB
> - -l 20：中频增益 20 dB
> - -a 1：启用天线供电（如果需要）