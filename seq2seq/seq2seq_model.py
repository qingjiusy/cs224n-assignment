import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
# torchtext的版本问题
from torchtext.data import Field, BucketIterato
# from torchtext.legacy.data import Field, BucketIterato

import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

# 加载 spaCy 的德语模型。"de" 是模型的标识符，表示你希望加载的是德语的预训练模型。
spacy_ger = spacy.load("de")
# spacy.load("en")：加载 spaCy 的英语模型。"en" 是模型的标识符，表示你希望加载的是英语的预训练模型
spacy_eng = spacy.load("en")

# 德语分词器
# 可以将"hello this world"变成["hello", "this", "world"]
def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

# 英语分词器
def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

# Field 是一个类，用于定义如何处理文本数据。它的参数指定了文本数据应该如何被预处理和转换。
# tokenize指定了用于将文本分词的函数
# lower指定是否将文本转换为小写
# init_token 指定了一个初始化标记（initial token），即在每个句子的开头添加的特殊标记。"<sos>" 表示句子的开始（start of sentence
# eos_token 指定了一个结束标记（end token），即在每个句子的结尾添加的特殊标记。"<eos>" 表示句子的结束（end of sentence）
german = Field(
    tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>"
    )

# 同样定义一个处理英语的Field类
english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)

# splits 方法用于将数据集分为训练集、验证集和测试集
# splits 方法是 torchtext 库中数据集类的一部分
# exts 参数指定了平行语料库的文件扩展名，即数据集中的源语言和目标语言文件扩展名
# fields 参数指定了应用于数据集的字段（Field）对象
# Multi30k 是 torchtext 库中提供的一个平行语料库数据集，主要用于机器翻译任务
train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

# vocabulary的定义
# build_vocab 方法是为指定的 Field 对象构建词汇表（vocabulary）。
# 当你调用 build_vocab 方法时，torchtext 会根据该 Field 关联的数据来构建词汇表。
# 因为 Field 对象（german）本身已经与特定的数据集字段（如德语或英语）相关联，所以不需要在 build_vocab 方法中再次指定语言。
# max_size表示词汇量表大小
# min_freq表示train_data中某个词加入词汇量表必须需要出现的次数，这里表示必须至少出现2次才会加入词汇量表
german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        # input_size：词汇的大小
        # embedding_size：每个单词的embedding size
        # num_layers: 许多层的lstm
        # p:Dropout的概率
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 嵌入层在自然语言处理（NLP）和其他相关领域中广泛使用，
        # 用于将离散的整数索引（如单词或者其他类别的标识符）映射到连续的低维向量空间中。
        self.embedding = nn.Embedding(input_size, embedding_size)

        # input_size：输入特征的大小，即输入序列的特征维度。
        # hidden_size：隐藏状态的大小，即 LSTM 层输出的特征维度
        # num_layers：LSTM 层的堆叠层数，默认为 1
        # dropout：如果非零，则在除了最后一层之外的 LSTM 层之间应用 dropout。默认为 0（不应用 dropout）
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape: (seq_length, N) 
        # seq_length：句子长度
        # where N is batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        # output 是 LSTM 层的所有时间步的输出，形状为 (a, b, c)，表示有 a 个时间步，batch_size 为 b，每个时间步的隐藏状态维度为 c。
        # (hn, cn) 是最后一个时间步的隐藏状态和细胞状态
        outputs, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        # output_size应该与input_size大小相同
        # 比如传入的是10000个单词，那么output_size就应该是这10000个单词对应概率
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
        # is 1 here because we are sending in a single word and not a sentence
        # 注意！这里不同于Encoder，Encoder这里是seq_length
        # 因为seq2seq在做的时候，encoder是可以将一整个句子直接encode，而decoder是一个一个decode
        
        # unsqueeze 方法用于在指定的维度上增加一个维度。
        # 具体来说，x.unsqueeze(0) 的作用是在 x 张量的第 0 维（也就是最外层的维度，通常是 batch 维度）上增加一个维度
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # 得到的hidden, cell用于下一个lstm 
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs)

        # predictions shape: (1, N, length_target_vocabulary) to send it to
        # loss function we want it to be (N, length_target_vocabulary) so we're
        # just gonna remove the first dim
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        # teacher_force_ratio表示我们在decode中用的真实比例
        # 比如这里表示使用的输入中 真实词:预测词 = 1:1
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        # 得到encoder中的hidden, cell用于decode
        hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        # 用于预测
        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            # 这里的output就是decoder中输出的predictions,需要存储起来
            output, hidden, cell = self.decoder(x, hidden, cell)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            # argmax(1) 中的 1 表示要沿着张量的第一个维度（通常是水平方向）进行操作，以找到每一行的最大值索引。
            # 这里的数字 1 是一个维度参数，用于指定操作应该沿着哪个维度执行。
            # argmax得到的是最大值索引
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            # 50%概率下一个输入不用这里的输出，而是使用正确的目标词汇
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


### We're ready to define everything we need for training our Seq2Seq model ###

# Training hyperparameters
num_epochs = 100
learning_rate = 0.001
batch_size = 64

# Model hyperparameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024  # Needs to be the same for both RNN's
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard to get nice loss plot
writer = SummaryWriter(f"runs/loss_plot")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    # 用sort_within_batch参数进行排序
    # 按照句子长度排序.所以相近长度的句子可以尽量被排在一起,这样可以减少padding,提高计算效率
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

# 具体选择模型
encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi["<pad>"]
# 在这个损失函数中，通过ignore_index=pad_idx参数告诉模型在计算损失时忽略填充的部分。
# 这在训练时非常有用，因为填充部分不应该对模型预测的准确性产生影响
# 当模型传入的索引是pad_idx时，交叉熵损失函数不会计算该位置的损失值
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")

    model.train()

    # batch训练
    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(inp_data, target)

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        # Back prop
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # Plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1


score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score*100:.2f}")