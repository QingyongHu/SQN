from os.path import join, exists, dirname, abspath
from SQN import Network
from tester_Semantic3D import ModelTester
from helper_ply import read_ply
from tool import Plot
from tool import DataProcessing as DP
from tool import ConfigSemantic3D as cfg
import tensorflow as tf
import numpy as np
import pickle, argparse, os, shutil


class Semantic3D:
    def __init__(self, labeled_point, gen_pseudo, retrain):
        self.name = 'Semantic3D'
        # set your dataset path here
        self.path = '/media/qingyong/QY/Semantic3D'
        self.label_to_names = {0: 'unlabeled', 1: 'man-made terrain', 2: 'natural terrain', 3: 'high vegetation',
                               4: 'low vegetation', 5: 'buildings', 6: 'hard scape', 7: 'scanning artefacts', 8: 'cars'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])

        # Original data path
        self.original_folder = join(self.path, 'original_data')
        self.full_pc_folder = join(self.path, 'original_ply')
        self.sub_pc_folder = join(self.path, 'input_{:.3f}'.format(cfg.sub_grid_size))

        # Following KPConv to do the train-validation split
        self.all_splits = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
        # self.all_splits = [0, 0, 4, 5, 3, 4, 1, 0, 1, 2, 3, 4, 2, 1, 0]

        self.use_val = True  # whether use validation set or not
        if self.use_val:
            self.val_split = 1
        else:
            self.val_split = 'no_val'

        # train files
        self.train_files = []
        self.val_files = []
        self.test_files = []
        cloud_names = [file_name[:-4] for file_name in os.listdir(self.original_folder) if file_name[-4:] == '.txt']
        for pc_name in cloud_names:
            if exists(join(self.original_folder, pc_name + '.labels')):
                self.train_files.append(join(self.sub_pc_folder, pc_name + '.ply'))
            else:
                self.test_files.append(join(self.full_pc_folder, pc_name + '.ply'))

        self.train_files = np.sort(self.train_files)
        self.test_files = np.sort(self.test_files)

        for i, file_path in enumerate(self.train_files):
            if self.all_splits[i] == self.val_split:
                self.val_files.append(file_path)

        self.train_files = np.sort([x for x in self.train_files if x not in self.val_files])

        self.gen_pseudo = gen_pseudo
        if gen_pseudo:
            self.train_files, self.test_files = self.test_files, self.train_files

        # initialize
        if '%' in labeled_point:
            r = float(labeled_point[:-1]) / 100
            self.num_with_anno_per_batch = max(int(cfg.num_points * r), 1)
        else:
            self.num_with_anno_per_batch = cfg.num_classes

        self.num_per_class = np.zeros(self.num_classes)
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.class_weight = {}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': [], 'test': []}
        self.input_names = {'training': [], 'validation': [], 'test': []}

        # Ascii files dict for testing
        self.ascii_files = {
            'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply': 'marketsquarefeldkirch4-reduced.labels',
            'sg27_station10_rgb_intensity-reduced.ply': 'sg27_10-reduced.labels',
            'sg28_Station2_rgb_intensity-reduced.ply': 'sg28_2-reduced.labels',
            'StGallenCathedral_station6_rgb_intensity-reduced.ply': 'stgallencathedral6-reduced.labels',
            'birdfountain_station1_xyz_intensity_rgb.ply': 'birdfountain1.labels',
            'castleblatten_station1_intensity_rgb.ply': 'castleblatten1.labels',
            'castleblatten_station5_xyz_intensity_rgb.ply': 'castleblatten5.labels',
            'marketplacefeldkirch_station1_intensity_rgb.ply': 'marketsquarefeldkirch1.labels',
            'marketplacefeldkirch_station4_intensity_rgb.ply': 'marketsquarefeldkirch4.labels',
            'marketplacefeldkirch_station7_intensity_rgb.ply': 'marketsquarefeldkirch7.labels',
            'sg27_station10_intensity_rgb.ply': 'sg27_10.labels',
            'sg27_station3_intensity_rgb.ply': 'sg27_3.labels',
            'sg27_station6_intensity_rgb.ply': 'sg27_6.labels',
            'sg27_station8_intensity_rgb.ply': 'sg27_8.labels',
            'sg28_station2_intensity_rgb.ply': 'sg28_2.labels',
            'sg28_station5_xyz_intensity_rgb.ply': 'sg28_5.labels',
            'stgallencathedral_station1_intensity_rgb.ply': 'stgallencathedral1.labels',
            'stgallencathedral_station3_intensity_rgb.ply': 'stgallencathedral3.labels',
            'stgallencathedral_station6_intensity_rgb.ply': 'stgallencathedral6.labels'}

        self.load_sub_sampled_clouds(cfg.sub_grid_size, labeled_point, retrain)
        for ignore_label in self.ignored_labels:
            self.num_per_class = np.delete(self.num_per_class, ignore_label)

    def load_sub_sampled_clouds(self, sub_grid_size, labeled_point, retrain):

        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        files = np.hstack((self.train_files, self.val_files, self.test_files))

        for i, file_path in enumerate(files):
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_pc_' + str(i) + ': ' + cloud_name)
            if file_path in self.val_files:
                cloud_split = 'validation'
            elif file_path in self.train_files:
                cloud_split = 'training'
            else:
                cloud_split = 'test'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # read ply with data
            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            if cloud_split == 'test':
                sub_labels = data['class']
            else:
                sub_labels = data['class']
                self.num_per_class += DP.get_num_class_from_label(sub_labels, self.num_classes)

            # ======================================== #
            #          Random Sparse Annotation        #
            # ======================================== #
            if cloud_split == 'training' and not self.gen_pseudo:
                if '%' in labeled_point:
                    # Randomly annotate x% of the points, keeping the rest of the points as unlabeled (i.e., label=0)
                    new_labels = np.zeros_like(sub_labels, dtype=np.int32)
                    num_pts = len(sub_labels)
                    r = float(labeled_point[:-1]) / 100
                    num_with_anno = max(int(num_pts * r), 1)
                    valid_idx = np.where(sub_labels)[0]
                    idx_with_anno = np.random.choice(valid_idx, num_with_anno, replace=False)
                    new_labels[idx_with_anno] = sub_labels[idx_with_anno]
                    sub_labels = new_labels
                else:
                    # Randomly annotate x point (e.g., 1pt setting) for each class
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

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_names[cloud_split] += [cloud_name]
            if cloud_split in ['training', 'validation'] or self.gen_pseudo:
                self.input_labels[cloud_split] += [sub_labels]

        # Get validation and test re_projection indices
        print('\nPreparing reprojection indices for validation and test')

        for i, file_path in enumerate(files):

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if file_path in self.val_files:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]

            # Test projection
            if file_path in self.test_files:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.test_proj += [proj_idx]
                self.test_labels += [labels]
        print('finished')
        return

    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
        elif split == 'test':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        # Reset possibility
        self.possibility[split] = []
        self.min_possibility[split] = []
        self.class_weight[split] = []

        for i, tree in enumerate(self.input_trees[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        if split == 'training' or (split == 'validation' and self.use_val):
            _, num_class_total = np.unique(np.hstack(self.input_labels[split]), return_counts=True)
            self.class_weight[split] += [np.squeeze([num_class_total / np.sum(num_class_total)], axis=0)]

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

                query_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]
                query_idx = DP.shuffle_idx(query_idx)

                # Collect points and colors
                queried_pc_xyz = points[query_idx]
                queried_pc_xyz[:, 0:2] = queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
                queried_pc_colors = self.input_colors[split][cloud_idx][query_idx]
                if split == 'test' or (split == 'validation' and not self.use_val):
                    if not self.gen_pseudo:
                        queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
                    else:
                        queried_pc_labels = self.input_labels[split][cloud_idx][query_idx]
                    queried_pt_weight = 1
                else:
                    queried_pc_labels = self.input_labels[split][cloud_idx][query_idx]
                    queried_pc_labels = np.array([self.label_to_idx[l] for l in queried_pc_labels])
                    queried_pt_weight = np.array([self.class_weight[split][0][n] for n in queried_pc_labels])

                dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight
                self.possibility[split][cloud_idx][query_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                if split == 'training':
                    unique_label_value = np.unique(queried_pc_labels)
                    if len(unique_label_value) <= 1:
                        i -= 1
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
                    yield (queried_pc_xyz,
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           query_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32),
                           xyz_with_anno.astype(np.float32),
                           labels_with_anno.astype(np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None], [None, 3], [None])
        return gen_func, gen_types, gen_shapes

    def get_tf_mapping(self):

        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, batch_xyz_anno,
                   batch_label_anno):
            batch_features = tf.map_fn(self.tf_augment_input, [batch_xyz, batch_features], dtype=tf.float32)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                neigh_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neigh_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neigh_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, batch_xyz_anno,
                           batch_label_anno]

            return input_list

        return tf_map

    @staticmethod
    def tf_augment_input(inputs):
        xyz = inputs[0]
        features = inputs[1]
        theta = tf.random_uniform((1,), minval=0, maxval=2 * np.pi)
        # Rotation matrices
        c, s = tf.cos(theta), tf.sin(theta)
        cs0 = tf.zeros_like(c)
        cs1 = tf.ones_like(c)
        R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
        stacked_rots = tf.reshape(R, (3, 3))

        # Apply rotations
        transformed_xyz = tf.reshape(tf.matmul(xyz, stacked_rots), [-1, 3])
        # Choose random scales for each example
        min_s = cfg.augment_scale_min
        max_s = cfg.augment_scale_max
        if cfg.augment_scale_anisotropic:
            s = tf.random_uniform((1, 3), minval=min_s, maxval=max_s)
        else:
            s = tf.random_uniform((1, 1), minval=min_s, maxval=max_s)

        symmetries = []
        for i in range(3):
            if cfg.augment_symmetries[i]:
                symmetries.append(tf.round(tf.random_uniform((1, 1))) * 2 - 1)
            else:
                symmetries.append(tf.ones([1, 1], dtype=tf.float32))
        s *= tf.concat(symmetries, 1)

        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = tf.tile(s, [tf.shape(transformed_xyz)[0], 1])

        # Apply scales
        transformed_xyz = transformed_xyz * stacked_scales

        noise = tf.random_normal(tf.shape(transformed_xyz), stddev=cfg.augment_noise)
        transformed_xyz = transformed_xyz + noise
        rgb = features[:, :3]
        stacked_features = tf.concat([transformed_xyz, rgb, features[:, 3:4]], axis=-1)
        return stacked_features

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        gen_function_test, _, _ = self.get_batch_gen('test')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)
        self.batch_test_data = self.batch_test_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)
        self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)
        self.test_init_op = iter.make_initializer(self.batch_test_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    parser.add_argument('--labeled_point', type=str, default='0.1%', help='0.1%/1%/10%/100%')
    parser.add_argument('--gen_pseudo', default=False, action='store_true', help='generate pseudo labels or not')
    parser.add_argument('--retrain', default=False, action='store_true', help='Re-training with pseudo labels or not')
    FLAGS = parser.parse_args()

    GPU_ID = FLAGS.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
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

    dataset = Semantic3D(FLAGS.labeled_point, FLAGS.gen_pseudo, FLAGS.retrain)
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
        tester.evaluate(model, dataset, FLAGS.gen_pseudo)
        shutil.rmtree('train_log') if exists('train_log') else None
    else:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.train_init_op)
            while True:
                a = sess.run(dataset.flat_inputs)
                pos = a[0]
                sub_pos1 = a[1]
                label = a[21]
                Plot.draw_pc_sem_ins(pos[0, :, :], label[0, :])
                Plot.draw_pc_sem_ins(sub_pos1[0, :, :], label[0, 0:np.shape(sub_pos1)[1]])
