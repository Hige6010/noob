import torch
import numpy as np
from sklearn.metrics import roc_curve, average_precision_score, precision_score, recall_score
from sklearn import metrics

#计算模型的分类准确率。
def calc_accuracy(y_softmax, y_true):
    predictions = y_softmax.argmax(dim=1)
    acc = (predictions == y_true).float().mean().item()#计算模型的分类准确率。
    return acc

#模型对 “高置信度样本” 的分类准确率
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
    #对筛选出的高置信度样本，先通过 np.argmax(..., axis=1) 得到每个样本的预测类别索引；
    #将预测类别索引与真实标签 y_true[top_samples] 对比，统计预测正确的样本数量 correct_predictions。
    correct_predictions = np.sum(np.argmax(y_softmax[top_samples], axis=1) == y_true[top_samples])
    accuracy = correct_predictions / num_top_samples
    return accuracy

# AURC, EAURC# AURC, EAURC衡量模型对分布外（OOD）样本鲁棒性
def calc_aurc_eaurc(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    #计算每个样本的最大置信度分数：对每个样本的 softmax 概率（形状为 (类别数,)），取最大值（即模型对该样本预测类别的 “把握程度”）
    softmax_max = np.max(softmax, 1)

    #将样本按置信度分数降序排序：把每个样本的 “置信度分数” 和 “预测正确性” 打包，通过 sorted 函数按置信度分数（x[0]）降序排列，得到排序后的样本元组列表。
    sort_values = sorted(zip(softmax_max[:], correctness[:]), key=lambda x:x[0], reverse=True)
    sort_softmax_max, sort_correctness = zip(*sort_values)
    #调用 coverage_risk 函数，计算每个覆盖比例下的风险（错误率）
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)
    #核心意义：AURC 和 EAURC 越小，说明模型在 “覆盖更多样本时，错误率增长越缓慢”，对分布外（OOD）样本的识别和鲁棒性越强。
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
    #提取风险列表的最后一个元素，即所有样本都被覆盖时的最终风险值（代表模型在 “完全覆盖所有样本” 时的错误率）
    r = risk_list[-1]
    #初始化风险 - 覆盖曲线下的面积（AURC）为 0。
    risk_coverage_curve_area = 0
    #计算最优风险 - 覆盖曲线的面积。该公式基于 “理想情况下，风险随覆盖比例的最优变化模式” 推导而来，用于衡量当前模型与 “最优鲁棒性” 的差距。
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        #将每个风险值乘以 “单个样本的覆盖比例权重（1/总样本数）”，并累加得到总面积。
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    #计算 EAURC：即当前 AURC 与最优风险 - 覆盖曲线面积的差值，用于衡量模型鲁棒性与 “理论最优” 的差距
    eaurc = risk_coverage_curve_area - optimal_risk_area
    return aurc, eaurc

# AUPR ERROR# AUPR ERROR计算AUCPR 错误（Average Precision Score 的误差）和 FPR@TPR95（真阳性率 95% 时的假阳性率），是评估模型在异常检测或分布外识别任务中性能的关键指标。
def calc_fpr_aupr(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)
    #假阳性率（FPR）、真阳性率（TPR）和对应的阈值。
    fpr, tpr, thresholds = metrics.roc_curve(correctness, softmax_max)
    #找到真阳性率（TPR）最接近95 % 的索引。
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    #提取该索引对应的假阳性率，即当模型识别出 95% 的真实正样本时，误将多少负样本识别为正样本。
    fpr_in_tpr_95 = fpr[idx_tpr_95]
    #计算平均精确率曲线下的面积（AUCPR）的误差。这里对correctness和softmax_max取负，是为了将 “正样本识别任务” 转换为 “异常检测任务” 的逻辑适配（将异常视为正样本）。
    aupr_err = metrics.average_precision_score(-1 * correctness + 1, -1 * softmax_max)
    return aupr_err, fpr_in_tpr_95

# ECE预期校准误差（Expected Calibration Error, ECE），用于衡量模型 “置信度与实际准确率的匹配程度”。
def calc_ece(softmax, label, bins=15):
    #将置信度范围（0 到 1）划分为bins个区间，用于分组统计模型的置信度和准确率。
    bin_boundaries = torch.linspace(0, 1, bins + 1)# steps=bins + 1：生成 bins + 1 个等距点，从而将 [0, 1] 划分为 bins 个连续子区间。
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # softmax = torch.tensor(softmax)
    # label = torch.tensor(label)

    #提取置信度与预测正确性
    #计算每个样本的最大置信度分数，并判断预测是否正确
    softmax_max, predictions = torch.max(softmax, 1)
    correctness = predictions.eq(label)

    ece = torch.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        '''
        gt(bin_lower.item())表示 “置信度大于区间下限”，
        le(bin_upper.item())表示 “置信度小于等于区间上限”
        标记哪些样本属于当前置信度区间。
        '''
        in_bin = softmax_max.gt(bin_lower.item()) * softmax_max.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = softmax_max[in_bin].mean()
            #计算 “平均置信度” 与 “实际准确率” 的绝对差，再乘以 “区间样本占比”，并累加到总 ECE 中。
            #对样本多的区间赋予更高的权重，确保 ECE 能反映模型在整体上的校准偏差。
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
     #置信度校准程度的关键指标。ECE 越小，说明模型的 “自信程度” 与 “实际表现” 越匹配，预测可靠性越高。
    return ece.item()

# NLL & Brier Score负对数似然（NLL）和 Brier 分数，是衡量模型 “概率预测质量” 的关键指标。
def calc_nll_brier(softmax, label, label_onehot):

    log_softmax = torch.log(softmax)
    softmax = np.array(softmax)
    label_onehot = np.array(label_onehot)
    #“模型概率预测与真实标签分布的均方误差”
    brier_score = np.mean(np.sum((softmax - label_onehot) ** 2, axis=1))
    nll = calc_nll(log_softmax, label)
    #Brier 分数越小：模型的概率预测与真实标签的分布差异越小，概率预测的 “校准性” 越好。
    return nll.item(), brier_score

# Calc NLL
def calc_nll(log_softmax, label):
    out = torch.zeros_like(label, dtype=torch.float)
    for i in range(len(label)):
        out[i] = log_softmax[i][label[i]]
    #NLL 越小：模型对真实标签的预测概率越高，说明概率分布与真实标签的匹配度越好。
    return -out.sum()/len(out)
