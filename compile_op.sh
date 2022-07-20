cd utils/nearest_neighbors
python setup.py install --home="."
cd ../../

cd utils/cpp_wrappers
sh compile_wrappers.sh
cd ../../

cd tf_ops/3d_interpolation
sh tf_interpolate_compile.sh
cd ../../
