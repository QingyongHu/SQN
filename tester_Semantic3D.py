from os import makedirs, system
from os.path import exists, join, dirname, abspath
from helper_ply import read_ply, write_ply
import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from tool import DataProcessing as DP


def log_string(out_str, log_out):
    log_out.write(out_str + '\n')
    log_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, model, dataset, restore_snap=None):
        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        # Add a softmax operation for predictions
        self.prob_logits = tf.nn.softmax(model.logits)
        self.test_probs = [np.zeros((l.data.shape[0], model.config.num_classes), dtype=np.float16)
                           for l in dataset.input_trees['test']]

        self.log_out = open('log_test_' + str(dataset.val_split) + '.txt', 'a')

    def evaluate(self, model, dataset, gen_pseudo=None, num_votes=100):

        # Smoothing parameter for votes
        test_smooth = 0.98

        # Initialise iterator with train data
        self.sess.run(dataset.test_init_op)

        if gen_pseudo:
            # Number of points per class in validation set
            val_proportions = np.zeros(model.config.num_classes, dtype=np.float32)
            i = 0
            for label_val in dataset.label_values:
                if label_val not in dataset.ignored_labels:
                    val_proportions[i] = np.sum(
                        [np.sum(labels == label_val) for labels in dataset.input_labels['test']])
                    i += 1

        # Test saving path
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'predictions')) if not exists(join(test_path, 'predictions')) else None
        makedirs(join(test_path, 'probs')) if not exists(join(test_path, 'probs')) else None

        #####################
        # Network predictions
        #####################

        step_id = 0
        epoch_id = 0
        last_min = -0.5

        while last_min < num_votes:

            try:
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'],)

                stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {model.is_training: False})
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, model.config.num_points,
                                                           model.config.num_classes])

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    inds = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                step_id += 1
                if not gen_pseudo:
                    log_string('Epoch {:3d}, step {:3d}. min possibility = {:.1f}'.format(epoch_id, step_id, np.min(
                        dataset.min_possibility['test'])), self.log_out)
                else:
                    stacked_probs = np.reshape(stacked_probs, [-1, model.config.num_classes])
                    pred = np.argmax(stacked_probs, axis=-1)
                    invalid_idx = np.where(stacked_labels == 0)[0]
                    labels_valid = np.delete(stacked_labels, invalid_idx)
                    pred_valid = np.delete(pred, invalid_idx)
                    labels_valid = labels_valid - 1
                    correct = np.sum(pred_valid == labels_valid)
                    acc = correct / float(len(labels_valid))
                    print('step' + str(step_id) + ' acc:' + str(acc))

            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_possibility['test'])
                log_string('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.log_out)

                if last_min + 4 < new_min:

                    print('Saving clouds')

                    if gen_pseudo:
                        # Show vote results (On subcloud so it is not the good values here)
                        log_string('\nConfusion on sub clouds', self.log_out)
                        confusion_list = []

                        num_test = len(dataset.input_labels['test'])

                        for i_test in range(num_test):
                            probs = self.test_probs[i_test]
                            for l_ind, label_value in enumerate(dataset.label_values):
                                if label_value in dataset.ignored_labels:
                                    probs = np.insert(probs, l_ind, 0, axis=1)

                            preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                            labels = dataset.input_labels['test'][i_test]

                            # Confs
                            confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]

                            # ==================================================== #
                            #          Generate pseudo labels for subclouds        #
                            # ==================================================== #

                            random_ratio = 0.05
                            trust_ratio = 0.01 / random_ratio
                            num_pts = len(preds)
                            
                            trust_preds = np.zeros_like(preds, dtype=np.int32)
                            random_num = max(int(num_pts * random_ratio), 1)
                            random_idx = np.random.choice(num_pts, random_num, replace=False)
                            
                            preds_random_selected = preds[random_idx]
                            probs_random_selected = probs[random_idx]
                            probs_random_selected_max_val = np.max(probs_random_selected, axis=1)
                            trust_idx_all = []
                            for i in range(dataset.num_classes):
                                ind_per_class = np.where(preds_random_selected == i)[0]  # idx belongs to class
                                num_per_class = len(ind_per_class)
                                if num_per_class > 0:
                                    trust_num = max(int(num_per_class * trust_ratio), 1)
                                    probs_max_val_per_class = probs_random_selected_max_val[ind_per_class]
                                    trust_pts_idx_per_class = probs_max_val_per_class.argsort()[-trust_num:][::-1]
                                    trust_idx_per_class = ind_per_class[trust_pts_idx_per_class]
                                    trust_idx_per_class = random_idx[trust_idx_per_class]
                                    trust_idx_all.append(trust_idx_per_class)
                            trust_idx_all = np.concatenate(trust_idx_all, axis=0)
                            trust_preds[trust_idx_all] = preds[trust_idx_all]

                            print(np.sum(preds[trust_idx_all] == labels[trust_idx_all]) / len(trust_idx_all))

                            name = dataset.input_names['test'][i_test] + '.ply'
                            write_ply(join(dirname(test_path), name), [trust_preds], ['pred'])

                        # Regroup confusions
                        C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)

                        # Remove ignored labels from confusions
                        for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
                            if label_value in dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)

                        # Rescale with the right number of point per class
                        C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                        # Compute IoUs
                        IoUs = DP.IoU_from_confusions(C)
                        m_IoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * m_IoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        log_string(s + '\n', self.log_out)
                        if gen_pseudo:
                            return

                    # Update last_min
                    last_min = new_min

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    files = dataset.test_files
                    i_test = 0
                    for i, file_path in enumerate(files):
                        # Get file
                        points = self.load_evaluation_points(file_path)
                        points = points.astype(np.float16)

                        # Reproject probs
                        probs = np.zeros(shape=[np.shape(points)[0], 8], dtype=np.float16)
                        proj_index = dataset.test_proj[i_test]

                        probs = self.test_probs[i_test][proj_index, :]

                        # Insert false columns for ignored labels
                        probs2 = probs
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs2 = np.insert(probs2, l_ind, 0, axis=1)

                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.uint8)

                        # Save plys
                        cloud_name = file_path.split('/')[-1]

                        # Save ascii preds
                        ascii_name = join(test_path, 'predictions', dataset.ascii_files[cloud_name])
                        np.savetxt(ascii_name, preds, fmt='%d')
                        log_string(ascii_name + 'has saved', self.log_out)
                        i_test += 1

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))
                    # creat submission files
                    base_dir = dirname(abspath(__file__))
                    results_path = join(base_dir, test_path, 'predictions')
                    system('cd %s && zip -r %s/reduced8.zip *-reduced.labels  && rm *-reduced.labels' % (
                        results_path, results_path))
                    system(
                        'cd %s && zip -r %s/semantic8.zip *.labels  && rm *.labels' % (results_path, results_path))
                    import sys
                    sys.exit()

                self.sess.run(dataset.test_init_op)
                epoch_id += 1
                step_id = 0
                continue
        return

    @staticmethod
    def load_evaluation_points(file_path):
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T
