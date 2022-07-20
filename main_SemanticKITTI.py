from tool import DataProcessing as DP
from tool import ConfigSemanticKITTI as cfg
from tool import Plot
from os.path import join, exists, abspath, dirname
from SQN import Network
from tester_SemanticKITTI import ModelTester
import tensorflow as tf
import numpy as np
import os, argparse, pickle, shutil


class SemanticKITTI:
    def __init__(self, test_id, labeled_point, gen_pseudo, retrain):
        self.name = 'SemanticKITTI'
        # set your dataset path here
        root_path = '/data/qy/Dataset'
        self.dataset_path = join(root_path, 'Semantic_KITTI', 'dataset', 'sequences_0.06')
        self.label_to_names = {0: 'unlabeled', 1: 'car', 2: 'bicycle', 3: 'motorcycle', 4: 'truck',
                               5: 'other-vehicle', 6: 'person', 7: 'bicyclist', 8: 'motorcyclist',
                               9: 'road', 10: 'parking', 11: 'sidewalk', 12: 'other-ground', 13: 'building',
                               14: 'fence', 15: 'vegetation', 16: 'trunk', 17: 'terrain', 18: 'pole',
                               19: 'traffic-sign'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])
        self.gen_pseudo = gen_pseudo
        self.retrain = retrain

        self.use_val = True  # whether use validation set or not
        self.val_split = '08'

        self.seq_list = np.sort(os.listdir(self.dataset_path))
        self.test_scan_number = str(test_id)
        self.train_list, self.val_list, self.test_list = DP.get_file_list(self.dataset_path,
                                                                          self.test_scan_number,
                                                                          self.gen_pseudo)

        self.train_list = DP.shuffle_list(self.train_list)
        self.val_list = DP.shuffle_list(self.val_list)

        self.possibility = []
        self.min_possibility = []
        self.num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                       240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                       9833174, 129609852, 4506626, 1168181])
        if '%' in labeled_point:
            r = float(labeled_point[:-1]) / 100
            self.num_with_anno_per_batch = max(int(cfg.num_points * r), 1)
        else:
            self.num_with_anno_per_batch = cfg.num_classes

        self.labeled_point = labeled_point

    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = int(len(self.train_list) / cfg.batch_size) * cfg.batch_size
            path_list = self.train_list
        elif split == 'validation':
            num_per_epoch = int(len(self.val_list) / cfg.val_batch_size) * cfg.val_batch_size
            cfg.val_steps = int(len(self.val_list) / cfg.batch_size)
            path_list = self.val_list
        elif split == 'test':
            num_per_epoch = int(len(self.test_list) / cfg.val_batch_size) * cfg.val_batch_size * 4
            path_list = self.test_list
            for test_file_name in path_list:
                points = np.load(test_file_name)
                self.possibility += [np.random.rand(points.shape[0]) * 1e-3]
                self.min_possibility += [float(np.min(self.possibility[-1]))]

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):
                if split != 'test':
                    cloud_ind = i
                    pc_path = path_list[cloud_ind]
                    pc, tree, labels = self.get_data(pc_path)
                    # crop a small point cloud
                    pick_idx = np.random.choice(len(pc), 1)
                    selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx)

                    if split == 'training':
                        unique_label_value = np.unique(selected_labels)
                        if len(unique_label_value) <= 1:
                            i -= 1
                            continue
                        else:
                            # ================================================================== #
                            #            Keep the same number of labeled points per batch        #
                            # ================================================================== #
                            idx_with_anno = np.where(selected_labels != self.ignored_labels[0])[0]
                            num_with_anno = len(idx_with_anno)
                            if num_with_anno > self.num_with_anno_per_batch:
                                idx_with_anno = np.random.choice(idx_with_anno, self.num_with_anno_per_batch,
                                                                 replace=False)
                            elif num_with_anno < self.num_with_anno_per_batch:
                                dup_idx = np.random.choice(idx_with_anno,
                                                           self.num_with_anno_per_batch - len(idx_with_anno))
                                idx_with_anno = np.concatenate([idx_with_anno, dup_idx], axis=0)
                            xyz_with_anno = selected_pc[idx_with_anno]
                            labels_with_anno = selected_labels[idx_with_anno]
                    else:
                        xyz_with_anno = selected_pc
                        labels_with_anno = selected_labels


                else:
                    cloud_ind = int(np.argmin(self.min_possibility))
                    pick_idx = np.argmin(self.possibility[cloud_ind])
                    pc_path = path_list[cloud_ind]
                    pc, tree, labels = self.get_data(pc_path)
                    selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx)

                    # update the possibility of the selected pc
                    dists = np.sum(np.square((selected_pc - pc[pick_idx]).astype(np.float32)), axis=1)
                    delta = np.square(1 - dists / np.max(dists))
                    self.possibility[cloud_ind][selected_idx] += delta
                    self.min_possibility[cloud_ind] = np.min(self.possibility[cloud_ind])
                    xyz_with_anno = selected_pc
                    labels_with_anno = selected_labels

                if True:
                    yield (selected_pc.astype(np.float32),
                           selected_labels.astype(np.int32),
                           selected_idx.astype(np.int32),
                           np.array([cloud_ind], dtype=np.int32),
                           xyz_with_anno.astype(np.float32),
                           labels_with_anno.astype(np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32)
        gen_shapes = ([None, 3], [None], [None], [None], [None, 3], [None])

        return gen_func, gen_types, gen_shapes

    def get_data(self, file_path):
        seq_id = file_path.split('/')[-3]
        frame_id = file_path.split('/')[-1][:-4]
        kd_tree_path = join(self.dataset_path, seq_id, 'KDTree', frame_id + '.pkl')

        # Read pkl with search tree
        with open(kd_tree_path, 'rb') as f:
            search_tree = pickle.load(f)
        points = np.array(search_tree.data, copy=False)
        # Load labels
        if int(seq_id) >= 11:
            labels = np.zeros(np.shape(points)[0], dtype=np.uint8)
        else:
            labeled_point = self.labeled_point
            label_path = join(self.dataset_path, seq_id, 'labels', frame_id + '.npy')
            labels = np.squeeze(np.load(label_path))
            if int(seq_id) != 8:
                # ======================================== #
                #          Random Sparse Annotation        #
                # ======================================== #
                if not self.gen_pseudo:
                    if '%' in labeled_point:
                        new_labels = np.zeros_like(labels, dtype=np.int32)
                        num_pts = len(labels)
                        r = float(labeled_point[:-1]) / 100
                        num_with_anno = max(int(num_pts * r), 1)
                        valid_idx = np.where(labels)[0]
                        idx_with_anno = np.random.choice(valid_idx, num_with_anno, replace=False)
                        new_labels[idx_with_anno] = labels[idx_with_anno]
                        labels = new_labels
                    else:
                        for i in range(self.num_classes):
                            ind_per_class = np.where(labels == i)[0]  # index of points belongs to a specific class
                            num_per_class = len(ind_per_class)
                            if num_per_class > 0:
                                num_with_anno = int(labeled_point)
                                num_without_anno = num_per_class - num_with_anno
                                idx_without_anno = np.random.choice(ind_per_class, num_without_anno, replace=False)
                                labels[idx_without_anno] = 0

                    # =================================================================== #
                    #            retrain the model with predicted pseudo labels           #
                    # =================================================================== #
                    if self.retrain:
                        pseudo_label_path = './test/sequences'
                        temp = np.load(join(pseudo_label_path, seq_id, 'predictions', frame_id + '.npy'))
                        pseudo_label = np.squeeze(temp)
                        pseudo_label_ratio = 0.01
                        pseudo_label[labels != 0] = labels[labels != 0]
                        labels = pseudo_label
                        self.num_with_anno_per_batch = int(cfg.num_points * pseudo_label_ratio)
        return points, search_tree, labels

    @staticmethod
    def crop_pc(points, labels, search_tree, pick_idx):
        # crop a fixed size point cloud for training
        center_point = points[pick_idx, :].reshape(1, -1)
        select_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]
        select_idx = DP.shuffle_idx(select_idx)
        select_points = points[select_idx]
        select_labels = labels[select_idx]
        return select_points, select_labels, select_idx

    @staticmethod
    def get_tf_mapping2():

        def tf_map(batch_pc, batch_label, batch_pc_idx, batch_cloud_idx, batch_xyz_anno,
                   batch_label_anno):
            features = batch_pc
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(DP.knn_search, [batch_pc, batch_pc, cfg.k_n], tf.int32)
                sub_points = batch_pc[:, :tf.shape(batch_pc)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:, :tf.shape(batch_pc)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_pc, 1], tf.int32)
                input_points.append(batch_pc)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_pc = sub_points

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx, batch_xyz_anno,
                           batch_label_anno]

            return input_list

        return tf_map

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

        map_func = self.get_tf_mapping2()

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
    parser.add_argument('--test_area', type=str, default='14', help='options: 11,12,13,14,15,16,17,18,19,20,21')
    parser.add_argument('--labeled_point', type=str, default='0.1%', help='0.1%/1%/10%/100%')
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
    dataset = SemanticKITTI(test_area, FLAGS.labeled_point, FLAGS.gen_pseudo, FLAGS.retrain)
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
                a = sess.run(dataset.flat_inputs)
                pos = a[0]
                sub_pos1 = a[1]
                label = a[17]
                Plot.draw_pc_sem_ins(pos[0, :, :], label[0, :])
                Plot.draw_pc_sem_ins(sub_pos1[0, :, :], label[0, 0:np.shape(sub_pos1)[1]])
