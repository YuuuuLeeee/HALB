#!/usr/bin/env python
# coding:utf-8
import copy

import numpy as np
import torch
import os


def _precision_recall_f1(right, predict, total):
    """
    :param right: int, the count of right prediction
    :param predict: int, the count of prediction
    :param total: int, the count of labels
    :return: p(precision, Float), r(recall, Float), f(f1_score, Float)
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f


def accuracy(epoch_predicts, epoch_labels, threshold):
    y_true = np.full((len(epoch_predicts), len(epoch_predicts[0])), 0)
    for i in range(len(epoch_labels)):
        for j in range(len(epoch_labels[i])):
            y_true[i][epoch_labels[i][j]] = 1
    y_predict = np.full((len(epoch_predicts), len(epoch_predicts[0])), 0)
    for i in range(len(epoch_predicts)):
        for j in range(len(epoch_predicts[i])):
            if epoch_predicts[i][j] > threshold:
                y_predict[i][j] = 1
    # compute accuracy
    count = 0
    for i in range(len(epoch_predicts)):
        p = sum(np.logical_and(y_true[i], y_predict[i]))
        q = sum(np.logical_or(y_true[i], y_predict[i]))
        count += p / q
    acc = count / y_true.shape[0]
    return acc


def HiP_HiR_HiF(epoch_predicts, epoch_labels, threshold, data_path):
    true_list = epoch_labels
    predict_list = []
    for i in range(len(epoch_predicts)):
        pre = []
        for j in range(len(epoch_predicts[i])):
            if epoch_predicts[i][j] > threshold:
                pre.append(j)
        predict_list.append(pre)
    label_hier = torch.load(os.path.join(data_path, 'slot.pt'))  # 导入标签层次结构
    path_dict = {}
    for s in label_hier:
        for v in label_hier[s]:  #
            path_dict[v] = s
    true_list_aug = true_list
    predict_list_aug = copy.deepcopy(predict_list)
    for i in range(len(predict_list)):
        for j in range(len(predict_list[i])):
            tem = predict_list[i][j]
            while(tem in path_dict):
                if path_dict[tem] not in predict_list_aug[i]:
                    predict_list_aug[i].append(path_dict[tem])
                tem = path_dict[tem]
    predict_aug_count = 0
    true_aug_count = 0
    predict_true_aug_count = 0
    for i in range(len(predict_list)):
        true_aug_count += len(true_list_aug[i])
        predict_aug_count += len(predict_list_aug[i])
        predict_true_aug_count += len(set(predict_list_aug[i]) & set(true_list_aug[i]))
    HiP = 1.0 * predict_true_aug_count / predict_aug_count if predict_aug_count>0 else 0.0
    HiR = 1.0 * predict_true_aug_count / true_aug_count if true_aug_count>0 else 0.0
    HiF = 2*HiP*HiR/(HiP+HiR) if (HiP+HiR)>0 else 0.0
    return HiP, HiR, HiF

def evaluate(epoch_predicts, epoch_labels, id2label, data_path=None, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'

    # acc,HIF,HIP,HIR
    if data_path:
        acc = accuracy(epoch_predicts, epoch_labels, threshold)
        HiP, HiR, HiF = HiP_HiR_HiF(epoch_predicts, epoch_labels, threshold, data_path)

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(id2label))] for _ in range(len(id2label))]
    right_count_list = [0 for _ in range(len(id2label))]
    gold_count_list = [0 for _ in range(len(id2label))]
    predicted_count_list = [0 for _ in range(len(id2label))]

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):  # 超过阈值0.5的就看作会贴这个标签
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        label = label + '_' + str(i)
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0
    if data_path:
        return {'precision': precision_micro,
                'recall': recall_micro,
                'micro_f1': micro_f1,
                'macro_f1': macro_f1,
                'accuracy': acc,
                'HiP':HiP,
                'HiR':HiR,
                'HiF':HiF,
                'full': [precision_dict, recall_dict, fscore_dict, right_count_list, predicted_count_list, gold_count_list]}
    else:
        return {'precision': precision_micro,
                'recall': recall_micro,
                'micro_f1': micro_f1,
                'macro_f1': macro_f1,
                'full': [precision_dict, recall_dict, fscore_dict, right_count_list, predicted_count_list,
                         gold_count_list]}


def evaluate_rcv_layer(epoch_predicts, epoch_labels, id2label, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(id2label))] for _ in range(len(id2label))]
    right_count_list = [0 for _ in range(len(id2label))]
    gold_count_list = [0 for _ in range(len(id2label))]
    predicted_count_list = [0 for _ in range(len(id2label))]

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        sample_predict_label_list = [id2label[i] for i in sample_predict_id_list]

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    right_total_layer1, predict_total_layer1, gold_total_layer1 = 0, 0, 0
    right_total_layer2, predict_total_layer2, gold_total_layer2 = 0, 0, 0
    right_total_layer3, predict_total_layer3, gold_total_layer3 = 0, 0, 0

    rcv1_layer1 = ['ccat', 'ecat', 'gcat', 'mcat']
    rcv1_layer2 = ['c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c21', 'c22', 'c23', 'c24', 'c31', 'c32',
                   'c33', 'c34', 'c41', 'c42', 'e11', 'e12', 'e13', 'e14', 'e21', 'e31', 'e41', 'e51', 'e61', 'e71',
                   'g15', 'gcrim', 'gdef', 'gdip', 'gdis', 'gent', 'genv', 'gfas', 'ghea', 'gjob', 'gmil', 'gobit',
                   'godd', 'gpol', 'gpro', 'grel', 'gsci', 'gspo', 'gtour', 'gvio', 'gvote', 'gwea', 'gwelf', 'm11',
                   'm12', 'm13', 'm14']
    rcv1_layer3 = ['c151', 'c152', 'c171', 'c172', 'c173', 'c174', 'c181', 'c182', 'c183', 'c311', 'c312', 'c313',
                   'c331', 'c411', 'e121', 'e131', 'e132', 'e141', 'e142', 'e143', 'e211', 'e212', 'e311', 'e312',
                   'e313', 'e411', 'e511', 'e512', 'e513', 'g151', 'g152', 'g153', 'g154', 'g155', 'g156', 'g157',
                   'g158', 'g159', 'm131', 'm132', 'm141', 'm142', 'm143']
    rcv1_layer4 = ['c1511']

    for i, label in id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]
        if label in rcv1_layer1:
            right_total_layer1 += right_count_list[i]
            gold_total_layer1 += gold_count_list[i]
            predict_total_layer1 += predicted_count_list[i]
        elif label in rcv1_layer2:
            right_total_layer2 += right_count_list[i]
            gold_total_layer2 += gold_count_list[i]
            predict_total_layer2 += predicted_count_list[i]
        else:
            right_total_layer3 += right_count_list[i]
            gold_total_layer3 += gold_count_list[i]
            predict_total_layer3 += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (
                                                                                                precision_micro + recall_micro) > 0 else 0.0

    precision_micro_layer1 = float(right_total_layer1) / predict_total_layer1 if predict_total_layer1 > 0 else 0.0
    recall_micro_layer1 = float(right_total_layer1) / gold_total_layer1
    micro_f1_layer1 = 2 * precision_micro_layer1 * recall_micro_layer1 / (
            precision_micro_layer1 + recall_micro_layer1) if (
                                                                     precision_micro_layer1 + recall_micro_layer1) > 0 else 0.0

    precision_micro_layer2 = float(right_total_layer2) / predict_total_layer2 if predict_total_layer2 > 0 else 0.0
    recall_micro_layer2 = float(right_total_layer2) / gold_total_layer2
    micro_f1_layer2 = 2 * precision_micro_layer2 * recall_micro_layer2 / (
            precision_micro_layer2 + recall_micro_layer2) if (
                                                                     precision_micro_layer2 + recall_micro_layer2) > 0 else 0.0

    precision_micro_layer3 = float(right_total_layer3) / predict_total_layer3 if predict_total_layer3 > 0 else 0.0
    recall_micro_layer3 = float(right_total_layer3) / gold_total_layer3
    micro_f1_layer3 = 2 * precision_micro_layer3 * recall_micro_layer3 / (
            precision_micro_layer3 + recall_micro_layer3) if (
                                                                     precision_micro_layer3 + recall_micro_layer3) > 0 else 0.0

    # l1-Macro-F1
    fscore_dict_l1 = [v for k, v in fscore_dict.items() if k in rcv1_layer1]
    macro_f1_layer1 = sum(fscore_dict_l1) / len(fscore_dict_l1)

    # l2-Macro-F1
    fscore_dict_l2 = [v for k, v in fscore_dict.items() if k in rcv1_layer2]
    macro_f1_layer2 = sum(fscore_dict_l2) / len(fscore_dict_l2)

    # l3-Macro-F1
    fscore_dict_l3 = [v for k, v in fscore_dict.items() if k not in rcv1_layer1 and k not in rcv1_layer2]
    macro_f1_layer3 = sum(fscore_dict_l3) / len(fscore_dict_l3)
    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'fscore_dict': fscore_dict,
            'l1_micro_f1': micro_f1_layer1,
            'l2_micro_f1': micro_f1_layer2,
            'l3_micro_f1': micro_f1_layer3,
            'l1_macro_f1': macro_f1_layer1,
            'l2_macro_f1': macro_f1_layer2,
            'l3_macro_f1': macro_f1_layer3}