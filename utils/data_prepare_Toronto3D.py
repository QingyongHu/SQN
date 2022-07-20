import pickle, os, sys, glob, getpass
import numpy as np
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree

# from plyfile import PlyData, PlyElement

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from tool import DataProcessing
from helper_ply import write_ply, read_ply
from tool import Plot

if __name__ == '__main__':
    dataset_name = 'Toronto_3D'
    grid_size = 0.04
    total_class = 9

    DP = DataProcessing(dataset_name, grid_size)
    nums_class = np.zeros(total_class, dtype=np.int32)

    for pc_path in glob.glob(join(DP.dataset_path, '*.ply')):
        print(pc_path)
        cloud_name = pc_path.split('/')[-1][:-4]

        # check if it has already calculated
        if exists(join(DP.sub_pc_folder, cloud_name + '_KDTree.pkl')):
            continue

        data = read_ply(pc_path)
        xyz = np.vstack((data['x'], data['y'], data['z'])).T
        xyz = (xyz - np.amin(xyz, axis=0)).astype(np.float32)  # normalize
        rgb = np.vstack((data['red'], data['green'], data['blue'])).T
        labels = (data['scalar_Label'])

        # remove invalid index
        valid_idx = np.squeeze(np.argwhere(~np.isnan(labels)), axis=1)
        xyz = (xyz[valid_idx]).astype(np.float32)
        rgb = rgb[valid_idx]
        labels = labels[valid_idx].astype(np.uint8)

        nums_class += DP.get_num_class_from_label(labels, total_class)
        DP.save_ply(cloud_name, xyz, rgb, labels, grid_size)

    # plot distribution
    Plot.plot_class_distribution(total_class, nums_class, dataset_name, False)
    # plot log_scale distribution
    Plot.plot_class_distribution(total_class, nums_class, dataset_name, True)
    print('total number of points', np.sum(nums_class))
    print('Statistics of original_labels:')
    print(nums_class)
    for idx, nums in enumerate(nums_class):
        print(idx, ':', nums)
