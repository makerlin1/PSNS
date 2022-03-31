# _*_ coding: utf-8 _*_
"""
Time:     2022-03-31 10:29
Author:   Haolin Yan(XiDian University)
File:     utils.py
"""

DEPTH2NUM = {"j": 10,
             "k": 11,
             "l": 12}

NUM2DEPTH = {10: "j",
             11: "k",
             12: "l"}


def pad_zero(vec, depth=12):
    while len(vec) < depth:
        vec.append(0)
    return vec


def Decimal_encoding(vec):
    """
    十进制编码
    :param vec:原编码
    :return:12维的特征(0~9)
    """
    depth = DEPTH2NUM[vec[0]]
    vec = vec[1:]
    sptr = 0
    eptr = sptr + 3
    result = []
    for i in range(depth):
        tuple_ele = [int(i) - 1 for i in vec[sptr:eptr - 1]]  # 获取三元组中的前两元组
        result.append(tuple_ele[0] + 3 * tuple_ele[1] + 1)
        sptr += 3
        eptr = sptr + 3
    return pad_zero(result)

# TODO:
# 编写解码函数
