#!/usr/bin/env python

import argparse
import numpy as np
import random

from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE (~1 Line)
    s = 1 / (1 + np.exp(-x))
    ### END YOUR CODE

    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models. For those unfamiliar with numpy notation, note 
    that a numpy ndarray with a shape of (x, ) is a one-dimensional array, which
    you can effectively treat as a vector with length x.

    实现中心词嵌入和外部词嵌入之间的简单 softmax 损失和梯度。这将是我们 word2vec 模型的构建模块。
    对于不熟悉 numpy 表示法的人，注意形状为 (x, ) 的 numpy ndarray 是一维数组，您可以有效地将其视为长度为 x 的向量。

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (tranpose of U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    参数：

    centerWordVec -- numpy ndarray,中心词的嵌入
                    形状为 (词向量长度, )
                    (pdf 讲义中的 v_c)
    outsideWordIdx -- 整数，外部词的索引
                    (pdf 讲义中的 o 或 u_o)
    outsideVectors -- 外部词向量
                    形状为 (词汇表中单词数, 词向量长度)
                    适用于词汇表中的所有单词 (pdf 讲义中的 U 的转置)
    dataset -- 负采样所需，这里不使用。

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)

    返回值：

    loss -- 简单 softmax 损失
    gradCenterVec -- 关于中心词向量的梯度
                    形状为 (词向量长度, )
                    (pdf 讲义中的 dJ / dv_c)
    gradOutsideVecs -- 关于所有外部词向量的梯度
                        形状为 (词汇表中单词数, 词向量长度)
                        (pdf 讲义中的 dJ / dU)
    """

    ### YOUR CODE HERE (~6-8 Lines)

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 

    gradOutsideVecs = np.zeros_like(outsideVectors)
    # 通过取向量点积并应用 softmax 获得 y_hat（即条件概率分布 p(O = o | C = c)）
    # np.dot(outsideVectors, centerWordVec)点积得到(词汇表中单词数,)这个形状的向量，里面的是分数
    # 然后取softmax来获得具体的概率
    y_hat = softmax(np.dot(outsideVectors, centerWordVec)) # (N,) N x 1
    # can also get y_hat in a single line: y_hat = softmax(outsideVectors @ centerWordVec)

    # for a single pair of words c and o, the loss is given by:
    # J(v_c, o, U) = -log P(O = o | C = c) = -log [y_hat[o]]
    # 用交叉熵损失（cross-entropy loss）来定义loss
    # outsideWordIdx：整数，真实外部词的索引
    # 真实外部词指的是在给定中心词时，实际与之共现的词。
    loss = -np.log(y_hat[outsideWordIdx])

    #计算梯度 
    # grad calc
    # generate the ground-truth one-hot vector, [..., 0, outsideWordIdx=1, 0, ...]
    y = np.zeros_like(y_hat)
    y[outsideWordIdx] = 1
    # can also get loss as -np.dot(y, np.log(y_hat))    
    
    # y_hat和y的形状：(N,)。outsideVectors的形状：(N, D)：(词汇表中单词数, 词向量长度)
    # 得到gradCenterVec的形状：(1, D)
    # gradCenterVec 是损失函数对中心词向量的梯度。它表示在当前中心词和外部词对下，如何调整中心词向量以减少损失。
    # 当预测的概率 y_hat 与真实分布 y 越接近时，差异 y_hat - y 就越小，梯度 gradCenterVec 也会相应变小。
    # 反之，当预测的概率 y_hat 与真实分布 y 差距越大时，差异 y_hat - y 就越大，梯度 gradCenterVec 也会相应变大，表示需要更大幅度地调整中心词向量。
    gradCenterVec = np.dot(y_hat - y, outsideVectors) # inner product results in a scalar
    # or gradCenterVec = np.dot(outsideVectors.T, y_hat - y)
    
    # 中心词向量 centerWordVec的形状：(D,)：(词向量长度,)
    # 外积运算得到一个矩阵
    # 得到的结果gradOutsideVecs的形状:(N, D) 的矩阵
    gradOutsideVecs = np.outer(y_hat - y, centerWordVec) # outer product results in a matrix
    # or gradOutsideVecs = np.dot((y_hat - y)[:, np.newaxis], centerWordVec[np.newaxis, :]) 
    
    # sanity check the dimensions
    assert gradCenterVec.shape == centerWordVec.shape
    assert gradOutsideVecs.shape == outsideVectors.shape  

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient


    负采样损失函数用于word2vec模型

    实现负采样损失和梯度,作为word2vec模型的一个基本构建块,针对中心词向量和外部词索引的词向量。这里的K是负样本的数量。

    注意：同一个词可能会被负采样多次。例如，如果一个外部词被采样两次，你需要对这个词的梯度进行双倍计算；如果被采样三次，则需要进行三倍计算，依此类推。

    参数/返回规范:与naiveSoftmaxLossAndGradient函数相同。
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    # getNegativeSamples是一个函数.用于从数据集中获取负样本的索引。
    # outsideWordIdx 是外部词的索引，K 是要采样的负样本数目。
    # indices：包含了一个正样本（即 outsideWordIdx）和 K 个负样本的索引列表
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE (~10 Lines)

    ### Please use your implementation of sigmoid in here.
    
    gradOutsideVecs = np.zeros(outsideVectors.shape)
    
    # 计算预测值和loss
    # 不是用之前的softmax去计算了,那样全部过outsideVectors一遍,即检查一整遍的词汇表,很浪费时间
    # 用sigmoid,只计算outsideWordIdx这一个位置
    y_hat = sigmoid(np.dot(outsideVectors[outsideWordIdx], centerWordVec))
    loss = -np.log(y_hat)
    
    # 减1是因为y_hat这个参数中只有outsideWordIdx的预测值,而outsideWordIdx的实际值应该就是1
    # !!!!!!!!!
    # 注意这里算gradCenterVec是偏差值(负的)和其他外部词或其他词的点积!
    gradCenterVec = np.dot(y_hat - 1, outsideVectors[outsideWordIdx])
    gradOutsideVecs[outsideWordIdx] = np.dot(y_hat - 1, centerWordVec)

    # Calculate the second term
    # 计算负采样的部分,其中的部分是随机抽的,假设他们不是外部词
    for i in range(K):
        w_k = indices[i+1]
        y_k_hat = sigmoid(-np.dot(outsideVectors[w_k], centerWordVec))
        # 累加loss
        loss += -np.log(y_k_hat)
        # 同时损失也要累加
        gradOutsideVecs[w_k] += np.dot(1.0 - y_k_hat, centerWordVec)
        gradCenterVec += np.dot(1.0 - y_k_hat, outsideVectors[w_k])

    ### END YOUR CODE
    # test
    # print(f"outsideWordIdx is :")
    # print(outsideWordIdx)
    # print(f"outsideVectors[outsideWordIdx]'s shape is :")
    # print(outsideVectors[outsideWordIdx].shape)
    # print(f"centerWordVec's shape is :")
    # print(centerWordVec.shape)

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (transpose of U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above

    参数：

    currentCenterWord：当前中心词的字符串
    windowSize：整数，上下文窗口大小
    outsideWords：长度不超过2*windowSize的字符串列表，表示外部词
    word2Ind：将单词映射到它们在词向量列表中的索引的字典
    centerWordVectors：中心词向量，以行的形式表示，形状为 (词汇表中的词数, 词向量长度)，对于词汇表中的所有单词（V 在pdf手册中）
    outsideVectors：外部词向量，以列的形式表示，形状为 (词汇表中的词数, 词向量长度)，对于词汇表中的所有单词（U 的转置 在pdf手册中）
    word2vecLossAndGradient：给定外部词索引的预测向量的损失和梯度函数，可以是你上面实现的两种损失函数之一。

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)

    返回：

    loss：跳字模型的损失函数值（J 在pdf手册中）
    gradCenterVec：关于中心词向量的梯度，形状为 (词向量长度, )
    gradOutsideVecs：关于所有外部词向量的梯度，形状为 (词汇表中的词数, 词向量长度)
    """
    # gradCenterVecs 和 gradOutsideVectors 分别初始化为与 centerWordVectors 和 outsideVectors 相同形状的零矩阵，用于累加每个外部词的梯度。
    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE (~8 Lines)

    # skip-gram model predicts outside words from the center word
    
    # get center word vec first from currentCenterWord

    # centerWordIdx 是当前中心词 currentCenterWord 在词汇表中的索引。
    # centerWordVec 是对应于中心词的向量表示，从 centerWordVectors 中获取。
    centerWordIdx = word2Ind[currentCenterWord]
    centerWordVec = centerWordVectors[centerWordIdx]

    # for 循环遍历 outsideWords 列表中的每个外部词
    # 对每个外部词，通过调用 word2vecLossAndGradient 函数计算损失 stepLoss、
    # 关于中心词向量的梯度 gradCenter 和关于外部词向量的梯度 gradOutside。
    for outsideWord in outsideWords:
        outsideWordIdx = word2Ind[outsideWord]
        stepLoss, gradCenter, gradOutside = word2vecLossAndGradient(centerWordVec,
                                                                    outsideWordIdx,
                                                                    outsideVectors,
                                                                    dataset)

        # 将每个外部词的损失值 stepLoss 累加到总损失 loss 中。
        loss += stepLoss
        gradCenterVecs[centerWordIdx] += gradCenter
        # 将每个外部词对所有外部词向量的梯度 gradOutside 累加到 gradOutsideVectors 上。
        gradOutsideVectors += gradOutside  

    ### END YOUR CODE
    
    return loss, gradCenterVecs, gradOutsideVectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset,
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad

def test_sigmoid():
    """ Test sigmoid function """
    print("=== Sanity check for sigmoid ===")
    assert sigmoid(0) == 0.5
    assert np.allclose(sigmoid(np.array([0])), np.array([0.5]))
    assert np.allclose(sigmoid(np.array([1,2,3])), np.array([0.73105858, 0.88079708, 0.95257413]))
    print("Tests for sigmoid passed!")

def getDummyObjects():
    """ Helper method for naiveSoftmaxLossAndGradient and negSamplingLossAndGradient tests """

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    dataset = type('dummy', (), {})()
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    return dataset, dummy_vectors, dummy_tokens

def test_naiveSoftmaxLossAndGradient():
    """ Test naiveSoftmaxLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for naiveSoftmaxLossAndGradient ====")
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec
    gradcheck_naive(temp, np.random.randn(3), "naiveSoftmaxLossAndGradient gradCenterVec")

    centerVec = np.random.randn(3)
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs
    gradcheck_naive(temp, dummy_vectors, "naiveSoftmaxLossAndGradient gradOutsideVecs")

def test_negSamplingLossAndGradient():
    """ Test negSamplingLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for negSamplingLossAndGradient ====")
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec
    gradcheck_naive(temp, np.random.randn(3), "negSamplingLossAndGradient gradCenterVec")

    centerVec = np.random.randn(3)
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs
    gradcheck_naive(temp, dummy_vectors, "negSamplingLossAndGradient gradOutsideVecs")

def test_skipgram():
    """ Test skip-gram with naiveSoftmaxLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")
    grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")
    grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient)

def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    test_sigmoid()
    test_naiveSoftmaxLossAndGradient()
    test_negSamplingLossAndGradient()
    test_skipgram()

if __name__ == "__main__":
    # 创建一个 ArgumentParser 对象，该对象用于处理命令行参数。description 参数提供了对这个命令行工具的简短描述。
    parser = argparse.ArgumentParser(description='Test your implementations.')
    # 添加一个名为 function 的命令行参数：
    # nargs='?' 表示这个参数是可选的。
    # type=str 指定这个参数的类型是字符串。
    # default='all' 设置了默认值为 'all'，如果用户没有提供这个参数，function 的值将会是 'all'。
    # help 描述了这个参数的用途。
    parser.add_argument('function', nargs='?', type=str, default='all',
                        help='Name of the function you would like to test.')

    args = parser.parse_args()

    # 输入命令行参数为sigmoid的话就运行test_sigmoid()
    if args.function == 'sigmoid':
        test_sigmoid()
    elif args.function == 'naiveSoftmaxLossAndGradient':
        test_naiveSoftmaxLossAndGradient()
    elif args.function == 'negSamplingLossAndGradient':
        test_negSamplingLossAndGradient()
    elif args.function == 'skipgram':
        test_skipgram()
    elif args.function == 'all':
        test_word2vec()
