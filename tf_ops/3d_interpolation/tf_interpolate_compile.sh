#/bin/bash
PYTHON=python3
CUDA_PATH=/usr/local/cuda
TF_PATH=$($PYTHON -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_PATH_LIB=$($PYTHON -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
PYTHON_VERSION=$($PYTHON -c 'import sys; print("%d.%d"%(sys.version_info[0], sys.version_info[1]))')
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I $TF_PATH -I $CUDA_PATH/include -I $TF_PATH/external/nsync/public -lcudart -L $CUDA_PATH/lib64/ -L $TF_PATH_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
