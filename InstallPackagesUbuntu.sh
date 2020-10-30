#!/bin/bash
sudo apt-get install g++ cmake libboost-program-options-dev libyaml-cpp-dev clang-tidy clang-format python3-matplotlib
pip3 install pytest
git clone https://github.com/pybind/pybind11.git 
cd pybind11 
git checkout v2.5.0
mkdir build 
cd build && cmake .. 
make -j8 
sudo make install
