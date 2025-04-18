<img src="./README/CQU-EIE.svg">
<h1 align="center">Design and Implementation of RF Fingerprint Extraction Technology for In-Car Bluetooth Devices</h1>

**Read this in other languages: [English](README.md), [中文](README_zh.md)**

## 🎈 Project

**Basic environment**: `Windows 11 Professional Workstation Edition 24H2`, `Anaconda3 2020.11 (Python 3.8.5 64-bit)`

1. Enter the directory of the project model

    ```bash
    cd .\model\
    ```

2. Then create the `conda` environment of the project

    ```bash
    conda env create -n bt python=3.10.16
    ```

3. Activate the environment

    ```bash
    conda activate bt
    ```

4. Change the `pip` source

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```

5. Download all project packages

    ```bash
    pip install -r requirements.txt
    ```
