from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import tf_util
import time
import Lovasz_losses_tf as L
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_interpolate import three_nn, three_interpolate


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


class Network:
    def __init__(self, dataset, config, retrain=False):
        flat_inputs = dataset.flat_inputs

        self.config = config
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d', time.gmtime())
                self.saving_path = self.saving_path + '_' + dataset.name
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['xyz'] = flat_inputs[:num_layers]
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
            self.inputs['features'] = flat_inputs[4 * num_layers]
            self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 2]
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 3]
            # self.inputs['batch_anno_inds'] = flat_inputs[4 * num_layers + 4]
            self.inputs['batch_xyz_anno'] = flat_inputs[4 * num_layers + 4]
            self.inputs['batch_label_anno'] = flat_inputs[4 * num_layers + 5]

            self.labels = self.inputs['labels']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            if dataset.name == 'S3DIS' or dataset.name == 'Semantic3D':
                self.loss_type = 'wce'  # sqrt, lovas
            else:
                self.loss_type = 'sqrt'  # wce, lovas
            self.class_weights = DP.get_class_weights(dataset.num_per_class, self.loss_type)
            self.Log_file = open('log_train_' + dataset.name + '_' + str(dataset.val_split) + '.txt', 'a')

        with tf.variable_scope('layers'):
            self.logits = self.inference(self.inputs, self.is_training)

        with tf.variable_scope('loss'):
            self.labels = self.inputs['batch_label_anno']

            with tf.device('/cpu:0'):
                self.labels = tf.cond(self.is_training,
                                      lambda: tf.concat([self.labels, self.labels], axis=0),
                                      lambda: self.labels)

            self.logits = tf.reshape(self.logits, [-1, config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)

            self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights)

        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=1)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def inference(self, inputs, is_training):

        d_out = self.config.d_out
        feature = inputs['features']
        batch_anno_xyz = inputs['batch_xyz_anno']  # annotated batch xyz

        with tf.device('/cpu:0'):
            feature = tf.cond(is_training,
                              lambda: tf.concat([feature, self.data_augment(feature)], axis=0),
                              lambda: feature)
            batch_anno_xyz = tf.cond(is_training,
                                     lambda: tf.concat([batch_anno_xyz, batch_anno_xyz], axis=0),
                                     lambda: batch_anno_xyz)

        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)

        # ###########################Encoder############################
        f_encoder_list = []
        f_interp = []
        for i in range(self.config.num_layers):
            xyz, neigh_idx, sub_idx = self.input_augment(is_training, inputs['xyz'][i], inputs['neigh_idx'][i],
                                                         inputs['sub_idx'][i])

            f_encoder_i = self.dilated_res_block(feature, xyz, neigh_idx, d_out[i],
                                                 'Encoder_layer_' + str(i), is_training)
            f_sampled_i = self.random_sample(f_encoder_i, sub_idx)
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)

        # ###########################Semantic Query############################
            idx = tf.py_func(DP.knn_search, [xyz, batch_anno_xyz, 3], tf.int32)
            neighbor_xyz = self.gather_neighbour(xyz, idx)
            xyz_tile = tf.tile(tf.expand_dims(batch_anno_xyz, axis=2), [1, 1, tf.shape(idx)[-1], 1])
            relative_xyz = xyz_tile - neighbor_xyz
            dist = tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=False)
            weight = tf.ones_like(dist) / 3.0
            # Trilinear interpolation
            interpolated_points = three_interpolate(tf.squeeze(f_encoder_i, axis=2), idx, weight)
            f_interp.append(interpolated_points)

        # Concatenation
        interpolated_points = tf.concat(f_interp, axis=-1)
        interpolated_points = tf.expand_dims(interpolated_points, axis=2)
        interpolated_points = tf_util.conv2d(interpolated_points, 256, [1, 1], 'if_1', [1, 1], 'VALID', True,
                                             is_training)
        interpolated_points = tf_util.conv2d(interpolated_points, 128, [1, 1], 'if_2', [1, 1], 'VALID', True,
                                             is_training)
        interpolated_points = tf_util.conv2d(interpolated_points, 64, [1, 1], 'if_3', [1, 1], 'VALID', True,
                                             is_training)

        f_layer_fc1 = tf_util.conv2d(interpolated_points, 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_fc3 = tf_util.conv2d(f_layer_fc2, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                     is_training, activation_fn=None)
        f_out = tf.squeeze(f_layer_fc3, [2])
        return f_out

    def train(self, dataset):
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.sess.run(dataset.train_init_op)
        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.accuracy]
                _, _, summary, l_out, probs, labels, acc = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:

                if dataset.use_val and self.training_epoch % 2 == 0:  # and self.training_epoch > 20
                    m_iou = self.evaluate(dataset)
                    if m_iou > np.max(self.mIou_list):
                        # Save the best model
                        snapshot_directory = join(self.saving_path, 'snapshots')
                        makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                        self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                    self.mIou_list.append(m_iou)
                    log_out('Best m_IoU of {} is: {:5.3f}'.format(dataset.name, max(self.mIou_list)), self.Log_file)
                else:
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', self.training_step)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # Update learning rate
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()

    @staticmethod
    def input_augment(is_training, xyz, neigh_idx, sub_idx):
        xyz_c = tf.concat([xyz, xyz], axis=0)
        neigh_idx_c = tf.concat([neigh_idx, neigh_idx], axis=0)
        sub_idx_c = tf.concat([sub_idx, sub_idx], axis=0)
        # , neigh_idx_o, sub_idx_o
        element = tf.cond(is_training,
                          lambda: [xyz_c, neigh_idx_c, sub_idx_c],
                          lambda: [xyz, neigh_idx, sub_idx])
        return element

    def data_augment(self, data):
        data_xyz = data[:, :, 0:3]
        batch_size = tf.shape(data_xyz)[0]
        aug_option = np.random.choice([0, 1, 2])
        if aug_option == 0:
            # mirror
            data_xyz = tf.stack([data_xyz[:, :, 0], -data_xyz[:, :, 1], data_xyz[:, :, 2]], 2)
        elif aug_option == 1:
            # rotation
            theta = 2 * 3.14592653 * np.random.rand()
            R = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
            R = tf.convert_to_tensor(R, dtype=tf.float32)
            data_xyz = tf.reshape(data_xyz, [-1, 3])
            data_xyz = tf.matmul(data_xyz, R)
            data_xyz = tf.reshape(data_xyz, [-1, self.config.num_points, 3])
        elif aug_option == 2:
            # jitter
            sigma = 0.01
            clip = 0.05
            jittered_point = tf.clip_by_value(sigma * np.random.randn(self.config.num_points, 3), -1 * clip, clip)
            jittered_point = tf.tile(tf.expand_dims(jittered_point, axis=0), [batch_size, 1, 1])
            data_xyz = data_xyz + tf.cast(jittered_point, tf.float32)
        if data.get_shape()[-1].value > 3:
            data_f = data[:, :, 3:]
            data_aug = tf.concat([data_xyz, data_f], axis=-1)
        else:
            data_aug = data_xyz
        data_aug_t = tf.transpose(data_aug, [0, 2, 1])
        data_aug_t = tf.reshape(data_aug_t, [-1, data.get_shape()[-1].value, self.config.num_points])
        att_activation = tf.layers.dense(data_aug_t, 1, activation=None, use_bias=False, name='channel_attention')
        att_activation = tf.transpose(att_activation, [0, 2, 1])
        att_scores = tf.nn.softmax(att_activation, axis=-1)
        data_aug = tf.multiply(data_aug, att_scores)
        return data_aug

    def evaluate(self, dataset):

        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy)
                stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})
                pred = np.argmax(stacked_prob, 1)
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n] + 0.1)
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        return mean_iou

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)

        if self.loss_type == 'lovas':
            logits = tf.reshape(logits, [-1, self.config.num_classes])  # [-1, n_class]
            probs = tf.nn.softmax(logits, axis=-1)  # [-1, class]
            labels = tf.reshape(labels, [-1])
            lovas_loss = L.lovasz_softmax(probs, labels, 'present')
            output_loss = output_loss + lovas_loss

        return output_loss

    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
        f_pc = tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                              activation_fn=None)
        shortcut = tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                  activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

        f_xyz = tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        # num_points = tf.shape(pc)[1]
        num_points = tf.shape(neighbor_idx)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg
