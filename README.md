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
