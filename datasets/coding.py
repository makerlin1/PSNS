# _*_ coding: utf-8 _*_
"""
Time:     2022-03-31 10:30
Author:   Haolin Yan(XiDian University)
File:     coding.py
"""
from .utils import Decimal_encoding

# TODO:
# 1) 拓展更多的编码方式，通过字典方式索引
# 2) 编写对应的解码函数
ENCODING_CLASSES = {"Decimal": Decimal_encoding}


def encode(x, Type="Decimal"):
    return ENCODING_CLASSES[Type](x)


