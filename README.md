# DDRL
Deep Developmental Reinforcement Learning

This source code is still in a research state, it has been used during 
my PhD thesis to develop several deep reinforcement learning agent in continuous environments (both in state and action).

<img src="environment/illustration.png" width=35% align="right" />

It contains : 
- 4 open-source and free environments using ODE (open dynamic engine) based on OpenAI/mujuco : acrobot, cartpole, half-cheetah and humanoid. We propose those environments because the free licence of Mujoco is not enough to perform hyperparameter meta-optimization on a cluster. However a lot of deep RL algorithms relies on a large number of hyperparameters.

- an implementation of PeNFAC with Caffe
```
Matthieu Zimmer and Paul Weng. Exploiting the sign of the advantage function to learn deterministic policies in continuous domains. In International Joint Conferences on Artificial Intelligence, August 2019.
```
- an implementation of DDPG with Caffe
```
Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., … Wierstra, D. (2015). Continuous control with deep reinforcement learning. arXiv Preprint arXiv:1509.02971.
```
- an implementation of NFAC(&lambda;)-V (extended with eligibility traces)
```
Matthieu Zimmer, Yann Boniface, and Alain Dutech. Neural fitted actor-critic. In ESANN – European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, April 2016.

Matthieu Zimmer. Developmental reinforcement learning. PhD thesis, University of Lorraine, January 2018.
```

- an implementation of developmental layers for NFAC(&lambda;)-V and DDPG
```
Matthieu Zimmer, Yann Boniface, and Alain Dutech. Developmental Reinforcement Learning through Sensorimotor Space Enlargement. Developmental reinforcement learning through sensorimotor space enlargement. In The 8th Joint IEEE International Conference on Development and Learning and on Epigenetic Robotics, September 2018.
```

- an implementation of CMA-ES with Caffe
```
Auger, A., & Hansen, N. (2005). A restart CMA evolution strategy with increasing population size. In Evolutionary Computation, 2005. The 2005 IEEE Congress on (Vol. 2, pp. 1769–1776).
```
- an synchronized and simplified implementation of A3C
```
Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., … Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning (pp. 1928–1937).
```
- an implementation of CACLA
```
Van Hasselt, H., & Wiering, M. A. (2007). Reinforcement learning in continuous action spaces. In Proceedings of the IEEE Symposium on Approximate Dynamic Programming and Reinforcement Learning (pp. 272–279). http://doi.org/10.1109/ADPRL.2007.368199
```
- an implementation of DPG (Determinist Policy Gradient)
```
Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014). Deterministic Policy Gradient Algorithms. Proceedings of the 31st International Conference on Machine Learning (ICML-14), 387–395.
```
- an implementation of SPG (Stochastic Policy Gradient)
```
Sutton, R. S., Mcallester, D., Singh, S., & Mansour, Y. (1999). Policy Gradient Methods for Reinforcement Learning with Function Approximation. In Advances in Neural Information Processing Systems 12, 1057–1063. http://doi.org/10.1.1.37.9714
```
- an implementation of Elastic Weight Constraint for Caffe
```
Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hassabis, D. (2017). Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13), 3521-3526.
```
- an implementation of inverting gradient strategy
```
Hausknecht, Matthew, and Peter Stone. "Deep reinforcement learning in parameterized action space." arXiv preprint arXiv:1511.04143 (2015).
```

Everything has been developed in C++.
The neural network library used is Caffe.

[![Demo](environment/video.gif)](https://www.youtube.com/watch?v=EzBmQsiUWBo)

## Install

Main dependencies : boost (>=1.54), caffe, ode(>=0.14).
However, it needs a modified version of Caffe : https://github.com/matthieu637/caffe.git
RAM+SWAP > 750 MB

### Archlinux 
```
yaourt -Sy boost ode openblas-lapack hdf5 protobuf google-glog gflags leveldb snappy lmdb cuda xz cmake gtest freeimage
cd any_directory_you_want
git clone https://github.com/matthieu637/ddrl
mkdir caffe_compilation 
cd caffe_compilation
cp ../ddrl/scripts/extern/PKGBUILD-CAFFE-CPU PKGBUILD
makepkg
sudo pacman -U caffe-ddrl-git-1-x86_64.pkg.tar.xz
cd ../ddrl/
./fullBuild.bash
```

### Ubuntu >= 14.04 (CPU Only)

Do not enable anaconda during compilation.
```
sudo apt-get update
#base
sudo apt-get install git libtool libboost-serialization-dev libboost-thread-dev libboost-system-dev libboost-program-options-dev libboost-filesystem-dev libtbb-dev libglew-dev python cmake libgtest-dev automake unzip libfreeimage-dev liblapacke-dev
#caffe
sudo apt-get install nvidia-cuda-dev nvidia-cuda-toolkit libprotobuf-dev libleveldb-dev libsnappy-dev protobuf-compiler libopenblas-dev libgflags-dev libgoogle-glog-dev liblmdb-dev libhdf5-serial-dev
#optional for developer
sudo apt-get install astyle cppcheck doxygen valgrind htop

cd any_directory_you_want
# gtest compilation
mkdir gtest
cp -r /usr/src/gtest/* gtest
cd gtest
cmake .
make -j4
sudo cp libgtest* /usr/lib/
cd ..

# caffe compilation
git clone https://github.com/matthieu637/caffe.git
mkdir caffe/build
cd caffe/build
cmake ../ -DBLAS=Open -DBUILD_python=OFF -DUSE_OPENCV=OFF -DCPU_ONLY=On -DCMAKE_INSTALL_PREFIX:PATH=/usr/local/
make -j4
sudo make install
cd ../..

# ode compilation is needed if official packages are under 0.14 (only required if fullbuild is called with --with-cpp)
mkdir ode
cd ode
wget https://bitbucket.org/odedevs/ode/downloads/ode-0.14.tar.gz
tar -xf ode-0.14.tar.gz
cd ode-0.14
./bootstrap
CFLAGS=-O2 CPPFLAGS=-O2 ./configure --prefix=/usr/local --enable-shared --enable-libccd --enable-double-precision --disable-asserts --disable-demos --with-drawstuff=none
make -j4
sudo make install
cd ../..

# then you can finnaly compile ddrl
git clone https://github.com/matthieu637/ddrl
cd ddrl
# to use DDRL with python (openai gym) the following build is enough
./fullBuild.bash

# if want to use ODE environments with c++ binaries
./fullBuild.bash --with-cpp
```

### Ubuntu LTS 18.04 with GPU (CUDA 10.0)
Install CUDA 10.0 (same as tensorflow)
```
#remove nvidia driver and default cuda
sudo apt-get -y remove nvidia-cuda-dev nvidia-cuda-toolkit nvidia-cuda-doc nvidia-cuda-gdb nvidia-utils-390 nvidia-driver-390 nvidia-utils-410 nvidia-driver-410
sudo apt-get autoremove
#optionally unload nvidia driver if you don't want to reboot (only for server without GUI)
sudo rmmod nvidia nvidia_uvm nvidia_modeset nvidia_drm

# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://202.121.180.31/cache/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-get update
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update
 
# Install NVIDIA driver
sudo apt-get install --no-install-recommends nvidia-driver-418
# Reboot. Check that GPUs are visible using the command: nvidia-smi
 
# Install development and runtime libraries (~4GB)
#cuda-10-0 needs nvidia-driver-418 and not nvidia-driver-410 ... (tensorflow.org seems outdated)
sudo apt-get install cuda-10-0 libcudnn7=7.5.1.10-1+cuda10.0 libcudnn7-dev=7.5.1.10-1+cuda10.0
sudo apt-get install nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0
sudo apt-get install libnvinfer-dev=5.1.5-1+cuda10.0 libnvinfer5=5.1.5-1+cuda10.0
 
#Protect those packages from being upgraded
sudo apt-mark hold libcudnn7 libcudnn7-dev libnvinfer-dev libnvinfer5
```

Needs a higher version of cmake than the one present in ubuntu 18.04
```
wget https://github.com/Kitware/CMake/releases/download/v3.13.5/cmake-3.13.5.tar.gz
tar xf cmake-3.13.5.tar.gz
cd cmake-3.13.5
./configure --prefix=$HOME/.root/cmake
make -j 4
make install
```

Follow the same installation as CPU only, except for caffe:
```
# caffe compilation
git clone https://github.com/matthieu637/caffe.git
mkdir caffe/build
cd caffe/build
~/.root/cmake/bin/cmake ../ -DBLAS=Open -DBUILD_python=OFF -DUSE_OPENCV=OFF -DCPU_ONLY=Off -DCMAKE_INSTALL_PREFIX:PATH=/usr/local/
make -j4
sudo make install
cd ../..
```

### Mac (CPU Only)
```
#if you run a version lower than sierra (example on mavericks)
#you need to install an up-to-date llvm version for c++11 features with :
#brew tap homebrew/versions
#brew install llvm38

#install brew
brew update
brew install cmake libtool findutils coreutils boost protobuf homebrew/science/hdf5 snappy leveldb gflags glog szip tbb lmdb gnu-sed
brew install --with-double-precision --with-shared ode

#caffe compilation
cd any_directory_you_want
git clone https://github.com/matthieu637/caffe.git
mkdir caffe/build
cd caffe/build
cmake ../ -DBLAS=Open -DBUILD_python=OFF -DUSE_OPENCV=OFF -DCPU_ONLY=On -DCMAKE_INSTALL_PREFIX:PATH=/usr/local/
make -j4
sudo make install
cd ../..

# then you can compile ddrl
git clone https://github.com/matthieu637/ddrl
cd ddrl
#if you want to use make to compile 
./fullBuild.bash
#or if you want to use Xcode projects
./fullBuild.bash xcode
```

### no access to sudo
if you don't have access to sudo, you can adapt the script under scripts/nosudo-install


## Usage with OpenAI Gym
Example to train PeNFAC on RoboschoolHalfCheetah-v1.

With a python virtual env
```
#create a python virtual environment with openai gym, roboschool
sudo apt-get install virtualenv
virtualenv ddrlvenv --python=python3
. ddrlvenv/bin/activate
pip install gym roboschool
```
or with anaconda:
```
. anaconda3/bin/activate
pip install gym roboschool
```

Goto gym directoy of ddrl and create a config.ini file with
```
[simulation]
total_max_steps=20000000
testing_each=20
#number of trajectories for testing
testing_trials=1


dump_log_each=50
display_log_each=100
save_agent_each=100000

library=ADAPT_PATH_TO_DDRL/agent/cacla/lib/libddrl-penfac.so
env_name=RoboschoolHalfCheetah-v1

[agent]
gamma=0.99
decision_each=1

#exploration
gaussian_policy=1
noise=0.2

#ADAM hyperparameters
momentum=0

#Neural network hyperparameters
hidden_unit_v=64:64
hidden_unit_a=64:64
#0 is linear, 1 is Leaky ReLU, 2 TanH, 3 ReLU
actor_output_layer_type=2
hidden_layer_type=1

#use online normalizer instead of batch norm
normalizer_type=2
batch_norm_actor=0
batch_norm_critic=0

#RL hyperparameters
alpha_a=0.0001
alpha_v=0.001
number_fitted_iteration=10
stoch_iter_critic=1
lambda=0.9
update_each_episode=5
stoch_iter_actor=30
beta_target=0.03

#fixed advanced setup
disable_cac=false
disable_trust_region=false
ignore_poss_ac=false
conserve_beta=true
gae=true
```
Then you can run this script within the virtual environment:
```
python run.py
```
In the current PeNFAC implementation:
- the learning process (all the forward/backward propagations on the neural networks) can be parallelized according to the value of the environment variable OMP_NUM_THREADS
- the process of collecting rollout on the environment is done in a synchronized way (i.e. with a single thread)

## Usage with C++ ODE environments

Do not enable anaconda in this case.
A .ini file is needed to describe the experience you want to run (neural network architecture, episodes, etc.).
```
#run the humanoid envionment with CMA-ES (debug version)
cd agent/cmaes/build/debug/
./humanoid --config cmaes-humanoid.ini

#run the humanoid environment with CMA-ES (release version with Xcode)
cd agent/cmaes/build/release/Release/
./humanoid --config ../cmaes-humanoid.ini

#run the humanoid envionment with CMA-ES (debug version + view)
cd agent/cmaes/build/debug/
./humanoid --config cmaes-humanoid.ini --view
```

The view option doesn't work on Mac because the GLUT thread need to be the main thread.

## Optimizing hyperparameters

See https://github.com/matthieu637/lhpo
