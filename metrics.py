import torch
import numpy as np
from sklearn.metrics import roc_curve, average_precision_score, precision_score, recall_score
from sklearn import metrics

def calc_accuracy(y_softmax, y_true):
    predictions = y_softmax.argmax(dim=1)
    acc = (predictions == y_true).float().mean().item()
    return acc

def calculate_top_accuracy(y_softmax, y_true, percent=0.9):
    y_softmax = np.array(y_softmax)
    y_true = np.array(y_true)
    # 计算每个样本的置信度分数
    confidence_scores = np.max(y_softmax, axis=1)
    # 根据置信度分数降序排序样本索引
    sorted_indices = np.argsort(-confidence_scores)
    # 计算置信度最高的90%样本的数量
    num_top_samples = int(percent * len(sorted_indices))
    # 选择置信度最高的90%样本
    top_samples = sorted_indices[:num_top_samples]
    # 计算准确率
    correct_predictions = np.sum(np.argmax(y_softmax[top_samples], axis=1) == y_true[top_samples])
    accuracy = correct_predictions / num_top_samples
    return accuracy

# AURC, EAURC
def calc_aurc_eaurc(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    sort_values = sorted(zip(softmax_max[:], correctness[:]), key=lambda x:x[0], reverse=True)
    sort_softmax_max, sort_correctness = zip(*sort_values)
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)

    return aurc, eaurc

# Calc coverage, risk
def coverage_risk(confidence, correctness):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)
        if correctness[i] == 0:
            risk += 1
        risk_list.append(risk / (i + 1))
    return risk_list, coverage_list

# Calc aurc, eaurc
def aurc_eaurc(risk_list):
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area
    return aurc, eaurc

# AUPR ERROR
def calc_fpr_aupr(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    fpr, tpr, thresholds = metrics.roc_curve(correctness, softmax_max)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]

    aupr_err = metrics.average_precision_score(-1 * correctness + 1, -1 * softmax_max)
    return aupr_err, fpr_in_tpr_95

# ECE
def calc_ece(softmax, label, bins=15):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # softmax = torch.tensor(softmax)
    # label = torch.tensor(label)

    softmax_max, predictions = torch.max(softmax, 1)
    correctness = predictions.eq(label)

    ece = torch.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = softmax_max.gt(bin_lower.item()) * softmax_max.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = softmax_max[in_bin].mean()

            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

# NLL & Brier Score
def calc_nll_brier(softmax, label, label_onehot):

    log_softmax = torch.log(softmax)
    softmax = np.array(softmax)
    label_onehot = np.array(label_onehot)
    brier_score = np.mean(np.sum((softmax - label_onehot) ** 2, axis=1))
    nll = calc_nll(log_softmax, label)
    return nll.item(), brier_score

# Calc NLL
def calc_nll(log_softmax, label):
    out = torch.zeros_like(label, dtype=torch.float)
    for i in range(len(label)):
        out[i] = log_softmax[i][label[i]]

    return -out.sum()/len(out)