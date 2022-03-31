# _*_ coding: utf-8 _*_
"""
Time:     2022-03-31 10:29
Author:   Haolin Yan(XiDian University)
File:     Decimal.py
"""
# _*_ coding: utf-8 _*_
"""
Time:     2022-03-31 8:33
Author:   Haolin Yan(XiDian University)
File:     dataset.py
"""
# _*_ coding: utf-8 _*_
"""
Time:     2022-03-28 21:22
Author:   Haolin Yan(XiDian University)
File:     Decimal.py.py
"""
from .coding import encode
import torch
from scipy.stats import rankdata
from torch.utils.data import Dataset
import json
import numpy as np

RANK_NAME = ['cplfw_rank',
             'market1501_rank',
             'dukemtmc_rank',
             'msmt17_rank',
             'veri_rank',
             'vehicleid_rank',
             'veriwild_rank',
             'sop_rank']


class DecimalDataset(Dataset):
    def __init__(self,
                 dataset_path="./data/CVPR_2022_NAS_Track2_train.json",
                 code_type="Decimal"):
        super(DecimalDataset, self).__init__()
        with open(dataset_path) as f:
            ds = json.load(f)
        self.ds = ds
        self.ds_len = len(ds)
        # 构建数据&标签
        data = []
        label = []
        for i in range(self.ds_len):
            arch = ds['arch' + str(i+1)]
            data.append(arch['arch'])
            rank = [int(arch[name]) for name in RANK_NAME]
            label.append(rank)
        self.label = label
        self.encode = lambda x: encode(x, code_type)
        self.data = [self.encode(d) for d in data]

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        return np.array(self.data[index]), np.array(self.label[index])

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return self.ds_len


def collate_fn(batch_data):
    """
    收集一个batch的标签，整合为[bsize, 8]的排序标签
    返回排名的rank
    e.g.
    原序列:[12,3,34,0]
    rank:[3,2,4,1]
    """
    X, y = zip(*batch_data)
    y_ = rankdata(np.array(y), axis=0)
    X = torch.FloatTensor(np.array(X))
    y_ = torch.FloatTensor(y_)
    return X, y_

