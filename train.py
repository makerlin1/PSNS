# _*_ coding: utf-8 _*_
"""
Time:     2022-03-31 10:37
Author:   Haolin Yan(XiDian University)
File:     train.py.py
"""
from datasets import DecimalDataset, collate_fn
from models import MLP
from utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch
import matplotlib.pyplot as plt


def train_one_epoch(epoch,
                    model,
                    dataset,
                    optim,
                    loss_fn,
                    batch_size,
                    dynamic_seq_size,
                    metric=None):
    model.train()
    device = get_net_device(model)
    train_loader = DataLoader(dataset,
                              batch_size=dynamic_seq_size,
                              shuffle=True,
                              collate_fn=collate_fn)
    nBatch = len(train_loader)
    losses = AverageMeter()
    with tqdm(total=nBatch,
              desc="Train Epoch #{}".format(epoch + 1),
              disable=True,
              ) as t:
        for batch_id, data in enumerate(train_loader):
            # 准备数据
            x_data = data[0].to(device)
            y_data = data[1].to(device)
            pred_y, _ = model(x_data)
            loss = loss_fn(pred_y.flatten(), y_data[:, 0])
            # 反向传播
            loss.backward()
            if (batch_id + 1) % batch_size == 1:
                # 更新参数
                optim.step()
                # 梯度清零
                optim.zero_grad()
            # Metric
            if metric is not None:
                # TODO: 编写指标
                print(metric(pred_y, y_data))
            losses.update(loss.item(), x_data.shape[0])
            t.set_postfix(
                {
                    "loss": losses.avg,
                }
            )
            t.update(1)
    return losses.avg


def validate(epoch,
             model,
             dataset,
             loss_fn,
             batch_size,
             metric=None):
    model.eval()
    device = get_net_device(model)
    val_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)
    nBatch = len(val_loader)
    losses = AverageMeter()
    with tqdm(total=nBatch,
              desc="Validate Epoch #{}".format(epoch + 1),
              ) as t:
        for batch_id, data in enumerate(val_loader()):
            x_data = data[0].to(device)
            y_data = data[1].to(device)
            pred_y, _ = model(x_data)
            loss = loss_fn(pred_y.flatten(), y_data[:, 0])
            if metric is not None:
                # TODO: 编写指标
                print(metric(pred_y, y_data))
            losses.update(loss.item(), x_data.shape[0])
            t.set_postfix(
                {
                    "loss": losses.avg,
                }
            )
            t.update(1)
    return losses.avg

Seq_size_list = [16]
Batch_size = 64
Epochs = 2048
model = MLP(12, 256, 1)
custom_dataset = DecimalDataset()
optim = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss()
model.train()
# ----------target device----------#
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     model = model.to(device)
#     cudnn.benchmark = True
#     print("Move model to %s" % device)

loss_list = []
for epoch in range(Epochs):
    dynamic_seq_size = Seq_size_list[epoch % len(Seq_size_list)]
    train_loss = train_one_epoch(epoch,
                                 model,
                                 custom_dataset,
                                 optim,
                                 loss_fn,
                                 batch_size=Batch_size,
                                 dynamic_seq_size=dynamic_seq_size)
    loss_list.append(train_loss)
    if (epoch + 1) % 100 == 0:
        print("epoch:%d, train loss:%f" % (epoch+1, train_loss))

plt.plot(loss_list)
plt.show()





