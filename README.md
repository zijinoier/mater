# MATER

MATER is a Multi-Agent in formation Training Environment for Reinforcement learning.

The training environment is inspired by  [libMultiRobotPlanning](https://github.com/whoenig/libMultiRobotPlanning) and uses pybind11 to communicate with python.  

## Install Pre-requirements

Tested on Ubuntu 16.04.

```bash
sudo apt-get install g++ cmake libboost-program-options-dev libyaml-cpp-dev clang-tidy clang-format python3-matplotlib
pip3 install pytest
git clone https://github.com/pybind/pybind11.git 
cd pybind11 
git checkout v2.5.0 #(optional)prefer using v2.5.0
mkdir build 
cd build && cmake .. 
make -j8 
sudo make install
```

or just simply run `InstallPackagesUbuntu.sh` in the root directory.

## Building

```bash
mkdir build
cd build
cmake ..
make -j16 && make install
```

Or just simply run `build.sh` in root directory.


## Run tests

```bash
python3 src/test_pybind.py 
```

For detailed API of `train_env`, please see

```python
import train_env as te
help(te)
```



