# _*_ coding: utf-8 _*_
"""
Time:     2022-03-31 10:43
Author:   Haolin Yan(XiDian University)
File:     tools.py
"""


def get_net_device(net):
    return net.parameters().__next__().device
