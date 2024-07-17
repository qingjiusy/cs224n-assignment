#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2020-2021: Homework 3
run.py: Run the dependency parser.
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""
from datetime import datetime
import os
import pickle
import math
import time
import argparse

from torch import nn, optim
import torch
from tqdm import tqdm

from parser_model import ParserModel
from utils.parser_utils import minibatches, load_and_preprocess_data, AverageMeter

parser = argparse.ArgumentParser(description='Train neural dependency parser in pytorch')
parser.add_argument('-d', '--debug', action='store_true', help='whether to enter debug mode')
args = parser.parse_args()

# -----------------
# Primary Functions
# -----------------
def train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005):
    """ Train the neural dependency parser.

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param output_path (str): Path to which model weights and results are written.
    @param batch_size (int): Number of examples in a single batch
    @param n_epochs (int): Number of training epochs
    @param lr (float): Learning rate
    """
    best_dev_UAS = 0


    ### YOUR CODE HERE (~2-7 lines)
    ### TODO:
    ###      1) Construct Adam Optimizer in variable `optimizer`
    ###      2) Construct the Cross Entropy Loss Function in variable `loss_func` with `mean`
    ###         reduction (default)
    ###
    ### Hint: Use `parser.model.parameters()` to pass optimizer
    ###       necessary parameters to tune.
    # 提示：使用`parser.model.parameters()`传递优化器所需的参数进行调整。
    ### Please see the following docs for support:
    ###     Adam Optimizer: https://pytorch.org/docs/stable/optim.html
    ###     Cross Entropy Loss: https://pytorch.org/docs/stable/nn.html#crossentropyloss
    optimizer = optim.Adam(parser.model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss(reduction='mean')

    ### END YOUR CODE

    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_UAS = train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size)
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            print("New best dev UAS! Saving model.")
            torch.save(parser.model.state_dict(), output_path)
        print("")


def train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size):
    """ Train the neural dependency parser for single epoch.

    Note: In PyTorch we can signify train versus test and automatically have
    the Dropout Layer applied and removed, accordingly, by specifying
    whether we are training, `model.train()`, or evaluating, `model.eval()`

    注意：在 PyTorch 中，我们可以表示训练与测试，
    并相应地自动应用和删除 Dropout 层，方法是指定我们是进行训练，`model.train()`，
    还是进行评估，`model.eval()`

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param optimizer (nn.Optimizer): Adam Optimizer
    @param loss_func (nn.CrossEntropyLoss): Cross Entropy Loss Function
    @param batch_size (int): batch size

    @return dev_UAS (float): Unlabeled Attachment Score (UAS) for dev data
    """
    parser.model.train() # Places model in "train" mode, i.e. apply dropout layer
    # math.ceil：向上取整函数
    # 计算小批量个数
    n_minibatches = math.ceil(len(train_data) / batch_size)
    # 这行代码创建了一个 AverageMeter 实例，并将其赋值给 loss_meter 变量。
    # AverageMeter 是一种常用的数据结构，通常在深度学习训练过程中用于跟踪和计算指标（例如损失、精度等）的平均值及其变化情况。
    loss_meter = AverageMeter()

    # tqdm 是一个 Python 库，用于显示进度条。
    # tqdm(total=(n_minibatches)) 创建一个进度条，总共有 n_minibatches 个小批量数据。
    # with 语句确保在退出代码块时正确关闭进度条。
    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            optimizer.zero_grad()   # remove any baggage in the optimizer
            loss = 0. # store loss for this batch here
            # 将 NumPy 数组 train_x 转换为 PyTorch 张量
            train_x = torch.from_numpy(train_x).long()
            # 返回非零元素的索引。对于分类任务，目标 train_y 通常是一个稀疏矩阵，使用 .nonzero() 获取其索引。
            # [1] 获取列索引
            # 然后将这些索引转换为 PyTorch 张量并转换为 torch.int64 数据类型。
            train_y = torch.from_numpy(train_y.nonzero()[1]).long()

            ### YOUR CODE HERE (~4-10 lines)
            ### TODO:
            ###      1) Run train_x forward through model to produce `logits`
            ###      2) Use the `loss_func` parameter to apply the PyTorch CrossEntropyLoss function.
            ###         This will take `logits` and `train_y` as inputs. It will output the CrossEntropyLoss
            ###         between softmax(`logits`) and `train_y`. Remember that softmax(`logits`)
            ###         are the predictions (y^ from the PDF).
            ###      3) Backprop losses
            ###      4) Take step with the optimizer
            # 1) 通过模型向前运行 train_x 以生成 `logits`
            # 2) 使用 `loss_func` 参数应用 PyTorch CrossEntropyLoss 函数。
            # 这将以 `logits` 和 `train_y` 作为输入。它将输出 softmax(`logits`) 和 `train_y` 之间的 CrossEntropyLoss。
            # 请记住，softmax(`logits`) 是预测（来自 PDF 的 y^）。
            # 3) 反向传播损失
            # 4) 与优化器一起迈出一步
            ### Please see the following docs for support:
            ###     Optimizer Step: https://pytorch.org/docs/stable/optim.html#optimizer-step

            # Forward pass: compute predicted logits
            logits = parser.model(train_x)
            
            # Compute loss
            loss = loss_func(logits, train_y)
            
            # Compute gradients of the loss w.r.t model parameters.
            loss.backward()
            
            # Take step with optimizer
            optimizer.step()

            ### END YOUR CODE
            # prog 是一个进度条对象，通常使用 tqdm 库创建。
            # update(1) 表示更新进度条的进度。参数 1 表示每调用一次 update，进度条的完成度增加一个单位（通常是一个批次）。
            # 例如，在训练循环中，每处理完一个小批量数据，就调用一次 prog.update(1) 来更新进度条的完成度。
            prog.update(1)

            # loss_meter 是一个用于计算和更新损失平均值的工具，可能是自定义的 AverageMeter 类或类似功能的实现。
            # update(loss.item()) 方法用于更新损失值的累积和统计信息。
            # 通常，在每次计算完损失后，调用 loss_meter.update(loss.item()) 来将当前批次的损失值加入到 loss_meter 中。
            loss_meter.update(loss.item())
    
    print ("Average Train Loss: {}".format(loss_meter.avg))

    print("Evaluating on dev set",)
    parser.model.eval() # Places model in "eval" mode, i.e. don't apply dropout layer
    dev_UAS, _ = parser.parse(dev_data)
    print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    return dev_UAS


if __name__ == "__main__":
    debug = args.debug

    assert (torch.__version__.split(".") >= ["1", "0", "0"]), "Please install torch version >= 1.0.0"

    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)

    start = time.time()

    # 一个pytorch的model
    model = ParserModel(embeddings)
    parser.model = model
    print("took {:.2f} seconds\n".format(time.time() - start))

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005)

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        parser.model.load_state_dict(torch.load(output_path))
        print("Final evaluation on test set",)
        parser.model.eval()
        UAS, dependencies = parser.parse(test_data)
        print("- test UAS: {:.2f}".format(UAS * 100.0))
        print("Done!")
