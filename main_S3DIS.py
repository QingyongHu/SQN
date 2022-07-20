from os.path import join, dirname, abspath, exists
from SQN import Network
from tester_S3DIS import ModelTester
from helper_ply import read_ply
from tool import ConfigS3DIS as cfg
from tool import DataProcessing as DP
from tool import Plot
import tensorflow as tf
import numpy as np
import time, pickle, argparse, glob, os, shutil


class S3DIS:
    def __init__(self, test_area_idx, labeled_point, retrain):
        self.name = 'S3DIS'
        # set your dataset path here
        root_path = '/data/qy/Dataset'
        self.path = join(root_path, self.name)
        self.label_to_names = {0: 'unlabel', 1: 'ceiling', 2: 'floor', 3: 'wall', 4: 'beam', 5: 'column', 6: 'window',
                               7: 'door', 8: 'table', 9: 'chair', 10: 'sofa', 11: 'bookcase', 12: 'board',
                               13: 'clutter'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        self.use_val = True  # whether use validation set or not
        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = np.sort(glob.glob(join(self.path, 'original_ply', '*.ply')))

        # initialize
        if '%' in labeled_point:
            r = float(labeled_point[:-1]) / 100
            self.num_with_anno_per_batch = max(int(cfg.num_points * r), 1)
        else:
            self.num_with_anno_per_batch = cfg.num_classes

        self.num_per_class = np.zeros(self.num_classes)
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size, labeled_point, retrain)
        for ignore_label in self.ignored_labels:
            self.num_per_class = np.delete(self.num_per_class, ignore_label)

    def load_sub_sampled_clouds(self, sub_grid_size, labeled_point, retrain):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class'] + 1

            # compute num_per_class in training set
            if cloud_split == 'training':
                self.num_per_class += DP.get_num_class_from_label(sub_labels, self.num_classes)

            # ======================================== #
            #          Random Sparse Annotation        #
            # ======================================== #
            if cloud_split == 'training':
                if '%' in labeled_point:
                    num_pts = len(sub_labels)
                    r = float(labeled_point[:-1]) / 100
                    num_with_anno = max(int(num_pts * r), 1)
                    num_without_anno = num_pts - num_with_anno
                    idx_without_anno = np.random.choice(num_pts, num_without_anno, replace=False)
                    sub_labels[idx_without_anno] = 0
                else:
                    for i in range(self.num_classes):
                        ind_per_class = np.where(sub_labels == i)[0]  # index of points belongs to a specific class
                        num_per_class = len(ind_per_class)
                        if num_per_class > 0:
                            num_with_anno = int(labeled_point)
                            num_without_anno = num_per_class - num_with_anno
                            idx_without_anno = np.random.choice(ind_per_class, num_without_anno, replace=False)
                            sub_labels[idx_without_anno] = 0

                # =================================================================== #
                #            retrain the model with predicted pseudo labels           #
                # =================================================================== #
                if retrain:
                    pseudo_label_path = './test'
                    temp = read_ply(join(pseudo_label_path, cloud_name + '.ply'))
                    pseudo_label = temp['pred']
                    pseudo_label_ratio = 0.01
                    pseudo_label[sub_labels != 0] = sub_labels[sub_labels != 0]
                    sub_labels = pseudo_label
                    self.num_with_anno_per_batch = int(cfg.num_points * pseudo_label_ratio)

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            # Plot.draw_pc_sem_ins(np.array(search_tree.data), sub_labels, 'S3DIS')

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            # print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices

        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                # print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        # Reset possibility
        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, tree in enumerate(self.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):  # num_per_epoch

                # Choose a random cloud
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get points from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                if len(points) < cfg.num_points:
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                queried_idx = DP.shuffle_idx(queried_idx)
                # Collect points and colors
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]
                queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]

                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][queried_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

                if split == 'training':
                    unique_label_value = np.unique(queried_pc_labels)
                    if len(unique_label_value) <= 1:
                        continue
                    else:
                        # ================================================================== #
                        #            Keep the same number of labeled points per batch        #
                        # ================================================================== #
                        idx_with_anno = np.where(queried_pc_labels != self.ignored_labels[0])[0]
                        num_with_anno = len(idx_with_anno)
                        if num_with_anno > self.num_with_anno_per_batch:
                            idx_with_anno = np.random.choice(idx_with_anno, self.num_with_anno_per_batch, replace=False)
                        elif num_with_anno < self.num_with_anno_per_batch:
                            dup_idx = np.random.choice(idx_with_anno, self.num_with_anno_per_batch - len(idx_with_anno))
                            idx_with_anno = np.concatenate([idx_with_anno, dup_idx], axis=0)
                        xyz_with_anno = queried_pc_xyz[idx_with_anno]
                        labels_with_anno = queried_pc_labels[idx_with_anno]
                else:
                    xyz_with_anno = queried_pc_xyz
                    labels_with_anno = queried_pc_labels

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32),
                           xyz_with_anno.astype(np.float32),
                           labels_with_anno.astype(np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None], [None, 3], [None])
        return gen_func, gen_types, gen_shapes

    @staticmethod
    def get_tf_mapping2():

        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, batch_xyz_anno,
                   batch_label_anno):
            batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, batch_xyz_anno,
                           batch_label_anno]

            return input_list

        return tf_map

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--labeled_point', type=str, default='0.1%', help='0.1%/1%/10%/100%')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    parser.add_argument('--gen_pseudo', default=False, action='store_true', help='generate pseudo labels or not')
    parser.add_argument('--retrain', default=False, action='store_true', help='Re-training with pseudo labels or not')
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    Mode = FLAGS.mode

    print('Settings:')
    print('Mode:', FLAGS.mode)
    print('Labeled_point', FLAGS.labeled_point)
    print('gen_pseudo', FLAGS.gen_pseudo)
    print('retrain', FLAGS.retrain)

    shutil.rmtree('__pycache__') if exists('__pycache__') else None
    if Mode == 'train':
        # shutil.rmtree('results') if exists('results') else None
        shutil.rmtree('train_log') if exists('train_log') else None
        for f in os.listdir(dirname(abspath(__file__))):
            if f.startswith('log_'):
                os.remove(f)

    test_area = FLAGS.test_area
    dataset = S3DIS(test_area, FLAGS.labeled_point, FLAGS.retrain)
    dataset.init_input_pipeline()

    if Mode == 'train':
        model = Network(dataset, cfg, FLAGS.retrain)
        model.train(dataset)
    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)
        chosen_snapshot = -1
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
        chosen_folder = logs[-1]
        snap_path = join(chosen_folder, 'snapshots')
        snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
        chosen_step = np.sort(snap_steps)[-1]
        chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset, FLAGS.gen_pseudo)
        shutil.rmtree('train_log') if exists('train_log') else None

    else:

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.train_init_op)
            while True:
                data_list = sess.run(dataset.flat_inputs)
                xyz = data_list[0]
                sub_xyz = data_list[1]
                label = data_list[21]
                Plot.draw_pc_sem_ins(xyz[0, :, :], label[0, :])
                Plot.draw_pc_sem_ins(sub_xyz[0, :, :], label[0, 0:np.shape(sub_xyz)[1]])
