import numpy as np
import glob, os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import read_ply
from tool import DataProcessing as DP

color_list = [[233, 229, 107],  # 'ceiling' .-> .yellow
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

if __name__ == '__main__':
    root_path = DP.get_dataset_root_path()
    base_dir = os.path.join(root_path, 'S3DIS', 'results')
    original_data_dir = os.path.join(root_path, 'S3DIS', 'original_ply')
    data_path = glob.glob(os.path.join(base_dir, '*.ply'))
    data_path = np.sort(data_path)

    test_total_correct = 0
    test_total_seen = 0
    gt_classes = [0 for _ in range(13)]
    positive_classes = [0 for _ in range(13)]
    true_positive_classes = [0 for _ in range(13)]

    for file_name in data_path:
        pred_data = read_ply(file_name)
        pred = pred_data['pred']
        pred -= 1
        original_data = read_ply(os.path.join(original_data_dir, file_name.split('/')[-1][:-4] + '.ply'))
        labels = original_data['class']
        points = np.vstack((original_data['x'], original_data['y'], original_data['z'])).T

        # =============================== #
        #            Visualize            #
        # =============================== #
        from tool import Plot
        Plot.draw_pc_sem_ins(points, labels, color_list)
        Plot.draw_pc_sem_ins(points, pred, color_list)

        correct = np.sum(pred == labels)
        print(str(file_name.split('/')[-1][:-4]) + '_acc:' + str(correct / float(len(labels))))
        test_total_correct += correct
        test_total_seen += len(labels)

        for j in range(len(labels)):
            gt_l = int(labels[j])
            pred_l = int(pred[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l == pred_l)

    iou_list = []
    for n in range(13):
        iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
        iou_list.append(iou)
    mean_iou = sum(iou_list) / 13.0

    acc_list = []
    for n in range(13):
        acc = true_positive_classes[n] / float(gt_classes[n])
        acc_list.append(acc)
    mean_acc = sum(acc_list) / 13.0

    print('mAcc value is :{}'.format(mean_acc))
    print('eval accuracy: {}'.format(test_total_correct / float(test_total_seen)))
    print('mean IOU:{}'.format(mean_iou))
    print(iou_list)
