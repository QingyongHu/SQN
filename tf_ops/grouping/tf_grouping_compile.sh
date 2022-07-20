#/bin/bash
PYTHON=python3
CUDA_PATH=/usr/local/cuda
TF_PATH=$($PYTHON -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_PATH_LIB=$($PYTHON -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
PYTHON_VERSION=$($PYTHON -c 'import sys; print("%d.%d"%(sys.version_info[0], sys.version_info[1]))')
$CUDA_PATH/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I $TF_PATH -I $CUDA_PATH/include -I $TF_PATH/external/nsync/public -lcudart -L $CUDA_PATH/lib64/ -L $TF_PATH_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0


