from os.path import join, exists, dirname, abspath
from helper_ply import write_ply
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle, gc, getpass
import colorsys, random, os, sys
import open3d as o3d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors


class ConfignuScenes:
    k_n = 16  # KNN
    num_layers = 4  # Number of layers
    num_points = 10240  # Number of input points
    num_classes = 16  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 48  # batch_size during training
    val_batch_size = 64  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256]  # feature dimension
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log_nuScenes'
    saving = True
    saving_path = None


class ConfigSemanticKITTI:
    k_n = 16  # KNN
    num_layers = 4  # Number of layers
    num_points = 4096 * 11  # Number of input points
    num_classes = 19  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 6  # batch_size during training
    val_batch_size = 18  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256]  # feature dimension
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log_KITTI'
    saving = True
    saving_path = None


class ConfigS3DIS:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 40960  # Number of input points
    num_classes = 13  # Number of valid classes
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 6  # batch_size during training
    val_batch_size = 12  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log_S3DIS'
    saving = True
    saving_path = None


class ConfigSemantic3D:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    num_classes = 8  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 4  # batch_size during training
    val_batch_size = 8  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log_Semantic3D'
    saving = True
    saving_path = None

    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_occlusion = 'none'
    augment_color = 0.8


class ConfigCity3D:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    num_classes = 13  # Number of valid classes
    sub_grid_size = 0.2  # preprocess_parameter

    batch_size = 4  # batch_size during training
    val_batch_size = 8  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log_city3D'
    saving = True
    saving_path = None


class ConfigScanNet:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 40960  # Number of input points
    num_classes = 20  # Number of valid classes
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 6  # batch_size during training
    val_batch_size = 12  # batch_size during validation and test
    train_steps = 1000  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # 2.0 noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log_ScanNet'
    saving = True
    saving_path = None


class ConfigToronto3D:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    num_classes = 8  # Number of valid classes
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 4  # batch_size during training
    val_batch_size = 8  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log_toronto3d'
    saving = True
    saving_path = None


class ConfigNPM3D:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    num_classes = 9  # Number of valid classes
    sub_grid_size = 0.08  # preprocess_parameter

    batch_size = 4  # batch_size during training
    val_batch_size = 8  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log_npm3d'
    saving = True
    saving_path = None


class ConfigDublin:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    num_classes = 11  # Number of valid classes
    sub_grid_size = 0.1  # preprocess_parameter

    batch_size = 4  # batch_size during training
    val_batch_size = 4  # batch_size during validation and test
    train_steps = 1000  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None


class ConfigDALES:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    num_classes = 8  # Number of valid classes
    sub_grid_size = 0.32  # preprocess_parameter

    batch_size = 4  # batch_size during training
    val_batch_size = 8  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None


class DataProcessing:
    def __init__(self, name, grid_size):
        root_path = self.get_dataset_root_path()
        self.dataset_path = join(root_path, name, 'original_data')
        self.original_pc_folder = join(dirname(self.dataset_path), 'original_ply')
        self.sub_pc_folder = join(dirname(self.dataset_path), 'input_{:.3f}'.format(grid_size))
        os.mkdir(self.original_pc_folder) if not exists(self.original_pc_folder) else None
        os.mkdir(self.sub_pc_folder) if not exists(self.sub_pc_folder) else None

    @staticmethod
    def get_dataset_root_path():
        import getpass
        import socket
        user_name = getpass.getuser()
        host_name = socket.gethostname()
        if user_name == 'qingyong' and host_name == 'qingyong-Desktop':
            root_path = '/data/Dataset/'
        elif user_name == 'qingyong' and host_name == 'qingyong-N95TP6':
            root_path = '/media/qingyong/32741D3B741D0371/CVPR2021_Evaluation/data/Dataset'
        elif user_name == 'root':
            root_path = '/root/workspace/data/Dataset'
        elif user_name == 'huqingyong':
            root_path = '/home/huqingyong/data3/Dataset'
        elif user_name == 'guo':
            root_path = '/home/guo/Qingyong/data/Dataset'
        elif user_name == 'qyjeffery':
            root_path = '/home/qyjeffery/data/Dataset'
        elif user_name == 'huqin':
            root_path = r'D://data//Dataset'
        elif user_name == 'qy-sysu':
            root_path = '/media/qy-sysu/data/Dataset'
        elif user_name == 'qy':
            root_path = '/home/qy/data/Dataset'
        elif user_name == 'qy2080':
            root_path = '/code/qy/data/Dataset'
        elif user_name == 'qytitian':
            root_path = '/home/qytitian/data/Dataset/'
        elif user_name == 'qyrtx':
            root_path = '/data2/qy/data/Dataset'
        elif host_name == 'lhrai80-lx':
            root_path = '/storage/local/qingyong/data/Dataset'
        elif host_name == 'qingyong-C9X299-PGF':
            root_path = '/media/qingyong/data/Dataset'
        elif user_name == 'qy1080':
            root_path = '/data/qy/Dataset'
        else:
            raise ValueError('undefined username or host name')
        return root_path

    @staticmethod
    def load_pc_semantic3d(filename):
        pc_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float16)
        pc = pc_pd.values
        xyz = pc[:, 0:3]
        rgb = pc[:, 4:7].astype(np.uint8)
        del pc
        gc.collect()
        return xyz, rgb

    @staticmethod
    def load_label_semantic3d(filename):
        label_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.uint8)
        cloud_labels = label_pd.values
        return cloud_labels

    @staticmethod
    def load_data_dales(pc_path):
        pc_pd = pd.read_csv(pc_path, header=None, delim_whitespace=True, dtype=np.float32)
        xyz = pc_pd.values
        xyz = (xyz - np.amin(xyz, axis=0))
        label_path = pc_path[:-4] + '.labels'
        label_pd = pd.read_csv(label_path, header=None, delim_whitespace=True, dtype=np.uint8)
        labels = label_pd.values
        labels = np.squeeze(labels)
        return xyz, labels

    @staticmethod
    def load_data_city3d(filename):
        from laspy.file import File
        if '.laz' in filename:
            data_label = File(filename, mode='r')

            xyz = np.vstack((data_label.x, data_label.y, data_label.z)).T
            rgb = np.vstack((data_label.red, data_label.green, data_label.blue)).T / 256
            rgb = rgb.astype(np.uint8)
            labels = data_label.classification.astype(np.uint8)
            # save memory
            del data_label
            gc.collect()
        else:
            pc_pd = pd.read_csv(filename, dtype=np.float32)
            pc = pc_pd.values

            xyz = pc[:, :3].astype(np.float32)
            rgb = pc[:, 3:6].astype(np.uint8)
            labels = pc[:, 6].astype(np.uint8)

        # Normalize 3D coordinates
        xyz = (xyz - np.amin(xyz, axis=0))
        xyz = xyz.astype(np.float32)
        labels = labels - 1
        return xyz, rgb, labels

    @staticmethod
    def load_pc_kitti(pc_path):
        scan = np.fromfile(pc_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, 0:3]  # get xyz
        return points

    @staticmethod
    def load_label_kitti(label_path, remap_lut):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = remap_lut[sem_label]
        return sem_label.astype(np.int32)

    @staticmethod
    def get_num_class_from_label(labels, total_class):
        num_pts_per_class = np.zeros(total_class, dtype=np.int32)
        # original class distribution
        val_list, counts = np.unique(labels, return_counts=True)
        for idx, val in enumerate(val_list):
            num_pts_per_class[val] += counts[idx]
        # for idx, nums in enumerate(num_pts_per_class):
        #     print(idx, ':', nums)
        return num_pts_per_class

    @staticmethod
    def get_nuscenes_file_list(dataset_path):
        train_file_list = np.array([join(dataset_path, 'train', 'velodyne', f) for f in
                                    np.sort(os.listdir(join(dataset_path, 'train', 'velodyne')))])
        val_file_list = np.array([join(dataset_path, 'val', 'velodyne', f) for f in
                                  np.sort(os.listdir(join(dataset_path, 'val', 'velodyne')))])
        test_file_list = np.array([join(dataset_path, 'test', 'velodyne', f) for f in
                                   np.sort(os.listdir(join(dataset_path, 'test', 'velodyne')))])
        return train_file_list, val_file_list, test_file_list

    @staticmethod
    def get_file_list(dataset_path, test_scan_num, gen_pesudo=None):
        seq_list = np.sort(os.listdir(dataset_path))
        train_file_list = []
        test_file_list = []
        val_file_list = []
        for seq_id in seq_list:
            seq_path = join(dataset_path, seq_id)
            pc_path = join(seq_path, 'velodyne')
            if seq_id == '08':
                val_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
                if seq_id == test_scan_num:
                    test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif (int(seq_id) >= 11 and seq_id == test_scan_num) or (gen_pesudo and seq_id == test_scan_num):
                test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                train_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

        train_file_list = np.concatenate(train_file_list, axis=0)
        val_file_list = np.concatenate(val_file_list, axis=0)
        test_file_list = np.concatenate(test_file_list, axis=0)
        return train_file_list, val_file_list, test_file_list

    def save_ply(self, cloud_name, xyz, rgb=None, labels=None, grid_size=0.1):
        print('Preparation of {:s}'.format(cloud_name))
        full_ply_path = join(self.original_pc_folder, cloud_name + '.ply')
        sub_ply_file = join(self.sub_pc_folder, cloud_name + '.ply')
        if rgb is not None:
            write_ply(full_ply_path, (xyz.astype(np.float32), rgb.astype(np.uint8), labels.astype(np.uint8)),
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
            # grid_sub_sampling to save memory and computation
            sub_xyz, sub_rgb, sub_labels = self.grid_sub_sampling(xyz, rgb, labels, grid_size)
            sub_rgb = sub_rgb / 255.0
            sub_labels = np.squeeze(sub_labels)
            write_ply(sub_ply_file, [sub_xyz, sub_rgb, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
        elif labels is not None:
            write_ply(full_ply_path, (xyz.astype(np.float32),
                                      labels.astype(np.uint8)), ['x', 'y', 'z', 'class'])
            sub_xyz, sub_labels = self.grid_sub_sampling(xyz, labels=labels, grid_size=grid_size)
            sub_labels = np.squeeze(sub_labels)
            write_ply(sub_ply_file, [sub_xyz, sub_labels], ['x', 'y', 'z', 'class'])
        else:
            write_ply(full_ply_path, xyz.astype(np.float32), ['x', 'y', 'z'])
            sub_xyz = self.grid_sub_sampling(xyz, grid_size=grid_size)
            write_ply(sub_ply_file, [sub_xyz], ['x', 'y', 'z'])
            labels = np.zeros(xyz.shape[0], dtype=np.uint8)

        search_tree = KDTree(sub_xyz, leaf_size=50)
        kd_tree_file = join(self.sub_pc_folder, cloud_name + '_KDTree.pkl')
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)

        proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
        proj_idx = proj_idx.astype(np.int32)
        proj_save = join(self.sub_pc_folder, cloud_name + '_proj.pkl')
        with open(proj_save, 'wb') as f:
            pickle.dump([proj_idx, labels], f)

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """
        CPP wrapper for a grid sub_sampling (method = barycenter for points and features
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: sub_sampled points, with features and/or labels depending of the input
        """

        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def get_class_weights(num_per_class, name='sqrt'):
        # # pre-calculate the number of points in each category
        frequency = num_per_class / float(sum(num_per_class))
        if name == 'sqrt' or name == 'lovas':
            ce_label_weight = 1 / np.sqrt(frequency)
        elif name == 'wce':
            ce_label_weight = 1 / (frequency + 0.02)
        else:
            raise ValueError('Only support sqrt and wce')
        return np.expand_dims(ce_label_weight, axis=0)


class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb):
        # only visualize a number of points to save memory
        num_pts = np.shape(pc_xyzrgb)[0]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            o3d.visualization.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])

        o3d.geometry.PointCloud.estimate_normals(pc)
        o3d.visualization.draw_geometries([pc], width=1000, height=1000)
        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, dataset=None):
        if dataset is None:
            ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=1)
        else:
            if dataset == 'S3DIS':
                plot_colors = [[0, 0, 0],  # 'unlabeled' .-> .black
                               [233, 229, 107],  # 'ceiling' .-> .yellow
                               [95, 156, 196],  # 'floor' .-> . blue
                               [179, 116, 81],  # 'wall'  ->  brown
                               [81, 163, 148],  # 'column'  ->  bluegreen
                               [241, 149, 131],  # 'beam'  ->  salmon
                               [77, 174, 84],  # 'window'  ->  bright green
                               [108, 135, 75],  # 'door'   ->  dark green
                               [79, 79, 76],  # 'table'  ->  dark grey
                               [41, 49, 101],  # 'chair'  ->  darkblue
                               [223, 52, 52],  # 'bookcase'  ->  red
                               [89, 47, 95],  # 'sofa'  ->  purple
                               [81, 109, 114],  # 'board'   ->  grey
                               [233, 233, 229],  # 'clutter'  ->  light grey
                               ]
            elif dataset == 'Semantic3D':
                plot_colors = [[0, 0, 0],  # invalid
                               [200, 200, 200],  # road
                               [0, 70, 0],  # grass
                               [0, 255, 0],  # tree
                               [255, 255, 0],  # bush
                               [255, 0, 0],  # buildings
                               [148, 0, 211],  # hardscape
                               [0, 255, 255],  # artefacts
                               [255, 8, 127]]  # cars
            elif dataset == 'Toronto3D':
                plot_colors = [[0, 0, 0],
                               [200, 200, 200],  # Road
                               [150, 34, 210],  # Road marking
                               [0, 251, 32],  # Natural
                               [255, 0, 0],  # Building
                               [75, 0, 175],  # Utility line
                               [0, 254, 250],  # Pole
                               [255, 17, 129],  # Car
                               [224, 163, 45]  # Fence
                               ]
            elif dataset == 'SemanticKITTI':
                plot_colors = [[0, 0, 0],  # unlabeled
                               [0, 0, 255],  # car
                               [245, 230, 100],  # bicycle
                               [150, 60, 30],  # motorcycle
                               [180, 30, 80],  # truck
                               [255, 0, 0],  # other-vehicle
                               [30, 30, 255],  # person
                               [200, 40, 255],  # bicyclist
                               [150, 60, 30],  # motorcyclist
                               [255, 0, 255],  # road
                               [255, 150, 255],  # parking
                               [218, 165, 32],  # sidewalk
                               [75, 0, 175],  # other-ground
                               [0, 200, 255],  # building
                               [50, 120, 255],  # fence
                               [0, 175, 0],  # vegetation
                               [0, 60, 135],  # trunk
                               [80, 240, 150],  # terrain
                               [150, 240, 255],  # pole
                               [0, 0, 255]  # traffic-sign
                               ]
            ins_colors = plot_colors

        # # only visualize a number of points to save memory
        # if plot_colors is not None:
        #     ins_colors = plot_colors
        # else:
        #     ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=1)

        ##############################
        sem_ins_labels = np.unique(pc_sem_ins)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            ### bbox
            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0]);
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1]);
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2]);
            zmax = np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins)
        return Y_semins

    @staticmethod
    def plot_class_distribution(x_input, y_input, dataset_name, use_log_scale=True):
        sns.set_style("whitegrid")
        # plot bar chart of class
        fig = plt.figure()
        fig.set_size_inches(16, 12)
        sns.barplot(np.arange(0, x_input, 1), y_input)
        tick_labels = ['class:' + (str(x)) for x in range(x_input)]
        plt.xticks(np.arange(x_input), tick_labels, rotation='vertical')
        if use_log_scale:
            plt.yscale("log")
            fig.savefig('class_distribution_log' + str(dataset_name) + '.png')
        else:
            fig.savefig('class_distribution' + str(dataset_name) + '.png')
        plt.close()

    @staticmethod
    def save_ply_o3d(data, save_name):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, 0:3])
        if np.shape(data)[1] == 3:
            o3d.io.write_point_cloud(save_name, pcd)
        elif np.shape(data)[1] == 6:
            if np.max(data[:, 3:6]) > 20:
                pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.)
            else:
                pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6])
            o3d.io.write_point_cloud(save_name, pcd)
        return

    @staticmethod
    def remove_invalid_pts(data, label):
        invalid_idx = np.where(label == 0)[0]
        data_valid = np.delete(data, invalid_idx, axis=0)
        label_valid = np.delete(label, invalid_idx)
        return data_valid, label_valid

    @staticmethod
    def sample4vis(data, label, num):
        idx = np.random.choice(len(label), num)
        sub_label = label[idx]
        sub_data = data[idx, :]
        return sub_data, sub_label
