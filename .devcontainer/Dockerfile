FROM ubuntu:18.04

RUN apt-get update \
    # Pre-requsities
    && apt-get install -y wget git build-essential gdb libxt-dev libx11-6 libx11-dev libxmu-dev libxpm-dev libxaw7-dev \
    #
    # Install CMake binary
    && cmake_version=3.17.1 \
    && mkdir ~/temp \
    && cd ~/temp \
    && wget https://github.com/Kitware/CMake/releases/download/v$cmake_version/cmake-$cmake_version-Linux-x86_64.sh \
    && mkdir /opt/cmake \
    && sh cmake-$cmake_version-Linux-x86_64.sh --prefix=/opt/cmake --skip-license \
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && cmake --version \
    #
    # C++
    && apt-get install -y clang \
    #
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y
