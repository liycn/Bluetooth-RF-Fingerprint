# 使用Ubuntu 20.04作为基础镜像
FROM ubuntu:20.04

# 设置环境变量，避免交互式安装时出现问题
ENV DEBIAN_FRONTEND=noninteractive

# 安装依赖项
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    g++ \
    libboost-all-dev \
    libgmp-dev \
    swig \
    python3-numpy \
    python3-mako \
    python3-sphinx \
    python3-lxml \
    doxygen \
    libfftw3-dev \
    libsdl1.2-dev \
    libgsl-dev \
    libqwt-qt5-dev \
    libqt5opengl5-dev \
    python3-pyqt5 \
    liblog4cpp5-dev \
    libzmq3-dev \
    python3-yaml \
    python3-click \
    python3-click-plugins \
    python3-zmq \
    python3-scipy \
    python3-gi \
    python3-cairo \
    && rm -rf /var/lib/apt/lists/*

# 克隆GNU Radio 3.7源码
RUN git clone https://github.com/gnuradio/gnuradio.git /gnuradio
WORKDIR /gnuradio
RUN git checkout maint-3.7

# 创建构建目录并编译
RUN mkdir build
WORKDIR /gnuradio/build
RUN cmake ..
RUN make -j$(nproc)
RUN make install
RUN ldconfig

# 设置默认命令
CMD ["gnuradio-companion"]
