from os.path import join, exists, dirname, abspath
import numpy as np
import glob
import sys

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from tool import DataProcessing
from tool import Plot

if __name__ == '__main__':
    dataset_name = 'Semantic3D'
    grid_size = 0.06
    total_class = 9

    DP = DataProcessing(dataset_name, grid_size)
    nums_class = np.zeros(total_class, dtype=np.int32)

    for pc_path in glob.glob(join(DP.dataset_path, '*.txt')):
        print(pc_path)
        cloud_name = pc_path.split('/')[-1][:-4]

        # check if it has already calculated
        if exists(join(DP.sub_pc_folder, cloud_name + '_KDTree.pkl')):
            continue

        # check if label exists
        label_path = pc_path[:-4] + '.labels'
        if exists(label_path):
            xyz, rgb = DP.load_pc_semantic3d(pc_path)
            labels = DP.load_label_semantic3d(label_path)
            xyz, rgb, labels = DP.grid_sub_sampling(xyz.astype(np.float32), rgb.astype(np.uint8), labels, 0.01)
            DP.save_ply(cloud_name, xyz, rgb, labels, grid_size)
        else:
            xyz, rgb = DP.load_pc_semantic3d(pc_path)
            labels = np.zeros(xyz.shape[0], dtype=np.uint8)
            DP.save_ply(cloud_name, xyz, rgb, labels, grid_size)
    # plot distribution
    Plot.plot_class_distribution(total_class, nums_class, False)
    # plot log_scale distribution
    Plot.plot_class_distribution(total_class, nums_class, True)
    print('total number of points', np.sum(nums_class))
    print('Statistics of original_labels:')
    print(nums_class)
    for idx, nums in enumerate(nums_class):
        print(idx, ':', nums)
