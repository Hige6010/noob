import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder

def get_sn(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num: view number
    :param alldata_len: number of samples
    :param missing_rate: Defined in section 3.2 of the paper
    :return: Sn:[0,1]matrix
    """
    one_rate = 1-missing_rate
    # 极端情况1：存在比例过低，每个样本至少保留1个视图
    if one_rate <= (1 / view_num):
        '''
        对于有 N 个不同类别的离散数据：
        为每个类别分配一个唯一的整数索引（如视图 0→0、视图 1→1、视图 2→2）；
        每个整数索引对应一个长度为 N 的二进制数组（“独热向量”）；
        数组中仅对应类别索引的位置为 1，其余位置均为 0（“独热” 即 “只有一个位置是热的（1）”）
        '''
        enc = OneHotEncoder()
        '''
        np.random.randint 是 NumPy 用于生成随机整数的函数。
        fit_transform 会先让编码器拟合数据，再将随机生成的视图索引数组转换为独热编码形式
        toarray() 会将其转换为密集的 NumPy 数组
        '''
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()#这一步会生成一个随机数组，每个元素是 [0, view_num-1] 之间的整数，代表每个样本被随机分配到某一个视图
        return view_preserve
    error = 1
    # 极端情况2：无缺失（存在比例为1）
    if one_rate == 1:
        #size=(all_data_len, view_num) 表示生成的数组有 all_data_len 行（对应样本数量）、view_num 列（对应视图数量）。
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        # 基础保留逻辑：为每个样本随机选择部分视图保留（稳定性）
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        #这一步是为了修正 “每个样本至少保留 1 个视图” 的约束
        one_num = view_num * alldata_len * one_rate - alldata_len# 目标“存在”的视图总数
        ratio = one_num / (view_num * alldata_len)
        #随机保留：按比例随机选择需要保留的视图（灵活性）
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int_)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int_))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int_)
        #只要 matrix_iter 或 view_preserve 中任意一个为1，结果就为True（表示 “最终保留该视图”）；只有两者都为0时才为False。
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int_)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    return matrix



def save_Sn(Sn,path):
    np.savetxt(path, Sn, delimiter=',')

def load_Sn(str_name):
    return np.loadtxt(str_name + '.csv', delimiter=',')


if __name__ == '__main__':

    sn = get_sn(6,2000,0.3)
    save_Sn(sn,'./sh.csv')

