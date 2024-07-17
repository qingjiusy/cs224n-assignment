import random
import torch
from torch.utils.data import Dataset
import argparse

"""
The input-output pairs (x, y) of the NameDataset are of the following form:

  x: Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  x: Where was Jacob Henry Studer born?⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

Using the PAD_CHAR characters in y before the ⁇[place] keeps the trainer from
optimizing the model to predict the question, "Where was...".

Note that the NameDataset should take the pretraining_dataset defined in run.py
as an input. This is to allow the vocab specification of the NameDataset to be
the same as that of the pretraining dataset.

You don't need to implement anything in NameDataset.
"""

class NameDataset(Dataset):
    def __init__(self, pretraining_dataset, data):
        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad
        self.itos = pretraining_dataset.itos 
        self.stoi = pretraining_dataset.stoi 
        self.block_size = pretraining_dataset.block_size
        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))

    def __len__(self):
        # returns the length of the dataset
        return len(self.data) - 1

    def __getitem__(self, idx):
        inp, oup = self.data[idx].split('\t')
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        x = x + self.PAD_CHAR*(self.block_size - len(x))
        y = self.PAD_CHAR*(len(inp)-1) + x[len(inp):]
        
        x = x[:-1]
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y


"""
[part e]

Write a class that yields examples of a simplified span corruption objective.
Do not change the signature of the __init__ or __getitem__ functions.

Make sure to implement the full spec for full credit -- we list below the
criteria that must be satisfied for a full implementation.

--------------
Vocabulary Specification

Your vocabulary is to be accessible via two dictionaries:
  self.stoi: a dictionary from characters in the vocabulary to indices of type
      int
  self.itos: a dictionary from indices of type int to characters in the
      vocabulary

Your vocabulary must have the following form: 

  Identifier 0 must be assigned to the unicode element u"\u25A1".
      This is the empty_square_character.
      Further, let self.PAD_CHAR = u"\u25A1"
  Identifier 1 must be assigned to the unicode element u"\u2047".
      This is the doublequestionmark character, which we'll use
      as a sentinel to represent that text is missing from the input
      Further, let self.MASK_CHAR = u"\u2047"
  Identifiers 2, ..., len(self.itos)-1 should be the sorted list of characters
      that appear in the data argument.

--------------
Masking Specification

The __getitem__ function takes an index and returns a data point (x, y) where
x and y are Long tensors of length self.block_size. x encodes the input
sequence, and y encodes the output sequence.

0. Use the idx argument of __getitem__ to retrieve the element of self.data
at the given index. We'll call the resulting data entry a document.

1. Randomly truncate the document to a length no less than 4 characters,
and no more than int(self.block_size*7/8) characters.

- IMPORTANT: You are free to decide how to perform this random truncation, but
make sure that the length is picked _randomly_ (every possible length from 4
to int(self.block_size*7/8) has a chance of being picked) for full credit.

__getitem__ 函数接受一个索引并返回一个数据点 (x, y)，其中 x 和 y 是长度为 self.block_size 的 Long 张量。
x 编码输入序列，y 编码输出序列。

使用 __getitem__ 的 idx 参数获取 self.data 中给定索引的元素。我们将得到的数据条目称为文档。

随机截断文档，使其长度不少于 4 个字符，且不超过 int(self.block_size * 7 / 8) 个字符。

重要提示：你可以自由决定如何执行这个随机截断，但请确保长度是随机选择
的（从 4 到 int(self.block_size * 7 / 8) 的每个可能长度都有被选择的机会），以获得全部积分。

2. Now, break the (truncated) document into three substrings:
    
    [prefix] [masked_content] [suffix]

  In other words, choose three strings prefix, masked_content and suffix
    such that prefix + masked_content + suffix = [the original document].
  The length of [masked_content] should be random, and 1/4 the length of the
    truncated document on average.

- IMPORTANT: You are free to decide how to perform this operation, but
make sure that the length is picked _randomly_ (has a chance of being more or
less than 1/4 the length of the truncated document) for full credit.

3. Rearrange these substrings into the following form:

    [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]
  
  This resulting string, denoted masked_string, serves as the output example.
  Here MASK_CHAR is the masking character defined in Vocabulary Specification,
    and [pads] is a string of repeated PAD_CHAR characters chosen so that the
    entire string is of length self.block_size.
  Intuitively, the [masked_content], a string, is removed from the document and
    replaced with MASK_CHAR (the masking character defined in Vocabulary
    Specification). After the suffix of the string, the MASK_CHAR is seen again,
    followed by the content that was removed, and the padding characters.

4. We now use masked_string to construct the input and output example pair. To
do so, simply take the input string to be masked_string[:-1], and the output
string to be masked_string[1:]. In other words, for each character, the goal is
to predict the next character in the masked string.

5. Making use of the vocabulary that you defined, encode the resulting input
and output strings as Long tensors and return the resulting data point.

2. 现在，将（截断后的）文档分成三个子字符串：

   [前缀] [被遮蔽的内容] [后缀]

   换句话说，选择三个字符串前缀、被遮蔽的内容和后缀，使得前缀 + 被遮蔽的内容 + 后缀 = [原始文档]。 
   [被遮蔽的内容] 的长度应该是随机的，平均为截断文档长度的 1/4。

   - 重要提示：你可以自由决定如何执行此操作，但请确保长度是随机选择的（有可能大于或小于截断文档长度的 1/4），以获得全部积分。

3. 将这些子字符串重新排列成以下形式：

   [前缀] MASK_CHAR [后缀] MASK_CHAR [被遮蔽的内容] [填充]

   这个生成的字符串，称为 `masked_string`，作为输出示例。
   这里 `MASK_CHAR` 是在词汇规范中定义的遮蔽字符，[填充] 是一串重复的 `PAD_CHAR` 字符，
   选择使整个字符串的长度为 `self.block_size`。
   直观上，[被遮蔽的内容] 是从文档中移除并用 `MASK_CHAR`（在词汇规范中定义的遮蔽字符）替换的。
   在字符串的后缀之后，`MASK_CHAR` 再次出现，后面跟着被移除的内容和填充字符。

4. 现在，我们使用 `masked_string` 来构造输入和输出示例对。
为此，只需将输入字符串设置为 `masked_string[:-1]`，将输出字符串设置为 `masked_string[1:]`。
换句话说，对于每个字符，目标是预测遮蔽字符串中的下一个字符。

5. 利用你定义的词汇，将结果输入和输出字符串编码为 Long 张量，并返回生成的数据点。

----------------
Here are some examples of input-output pairs (x, y):

  x: Khatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: hatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

  x: Jaco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: aco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

  x: John Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: ohn Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□


"""
class CharCorruptionDataset(Dataset):
    def __init__(self, data, block_size):
        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars 
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n')

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # TODO [part e]: see spec above
        doc = self.data[idx]
        
        # 1. Randomly truncate the document to a length no less than 4 characters,
        # and no more than int(self.block_size*7/8) characters.
        import random
        random_length = random.randint(4, int(self.block_size*7/8))
        truncated_doc = doc[:random_length]
        
        # 2. Now, break the (truncated) document into three substrings: [prefix] [masked_content] [suffix]
        #
        # In other words, choose three strings prefix, masked_content and suffix
        #  such that prefix + masked_content + suffix = [the original document].
        #  The length of [masked_content] should be random, and 1/4 the length of the
        #  truncated document on average.
        #
        # - IMPORTANT: You are free to decide how to perform this operation, but
        # make sure that the length is picked _randomly_ (has a chance of being more or
        # less than 1/4 the length of the truncated document) for full credit.

        # Take the length of the (randomly truncated) document.
        truncated_doc_length = len(doc)
        
        # Add a random perturbation to the length of the masked content with randint(). 
        # Just pick the right bounds, for example, -1/8 and 1/8 of the truncated length
        random_perturbation = random.randint(int(-0.125*truncated_doc_length), int(0.125*truncated_doc_length))

        # Compute expected_mask_len by using 1/4 of the truncated document length 
        # and adding a perturbation.
        expected_mask_len = int(0.25*truncated_doc_length + random_perturbation)
        # This gets you a masked length with minimum length of 1/8 of truncated length and 
        # maximum length of 3/8 of the truncated length, and the average is 1/4.   
        
        # randomly determine where to start the masked portion somewhere between the start 
        # of the truncated document and the last index such that there are as many characters 
        # left in the sequence as we determined the mask length should be
        start_mask = random.randint(0, truncated_doc_length - expected_mask_len)
        
        masked_content = truncated_doc[start_mask:start_mask+expected_mask_len]
        prefix = truncated_doc[:start_mask]
        suffix = truncated_doc[len(prefix)+expected_mask_len:]
        
        # 3. Rearrange these substrings into the following form:
        #
        #   [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]
        # 
        # This resulting string, denoted masked_string, serves as the output example.
        # Here MASK_CHAR is the masking character defined in Vocabulary Specification,
        #   and [pads] is a string of repeated PAD_CHAR characters chosen so that the
        #   entire string is of length self.block_size.
        # Intuitively, the [masked_content], a string, is removed from the document and
        #   replaced with MASK_CHAR (the masking character defined in Vocabulary
        #   Specification). After the suffix of the string, the MASK_CHAR is seen again,
        #   followed by the content that was removed, and the padding characters.
        
        masked_string = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_content
        masked_string += self.PAD_CHAR*(self.block_size - len(masked_string))
        
        # We now use masked_string to construct the input and output example pair. To
        # do so, simply take the input string to be masked_string[:-1], and the output
        # string to be masked_string[1:]. In other words, for each character, the goal is
        # to predict the next character in the masked string.
        
        input = masked_string[:-1]
        output = masked_string[1:]
        
        # Making use of the vocabulary that you defined, encode the resulting input
        # and output strings as Long tensors and return the resulting data point.
                
        x = torch.tensor([self.stoi[c] for c in input], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in output], dtype=torch.long)
        return x, y
        # raise NotImplementedError

"""
Code under here is strictly for your debugging purposes; feel free to modify
as desired.
"""
if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('dataset_type', help="Type of dataset to sample from."
            "Options: namedata, charcorruption.",
            choices=["namedata", "charcorruption"])
    args = argp.parse_args()

    if args.dataset_type == 'namedata':
        # Even if it hasn't been implemented, we use it to define the vocab
        corruption_dataset = CharCorruptionDataset(open('wiki.txt').read(), 128) 
        # Make the name dataset
        name_dataset = NameDataset(corruption_dataset,
            open('birth_places_train.tsv').read())
        for _, example in zip(range(4), name_dataset):
            x, y = example
            print('x:', ''.join([name_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([name_dataset.itos[int(c)] for c in y]))
        pass
    elif args.dataset_type == 'charcorruption':
        corruption_dataset = CharCorruptionDataset(open('wiki.txt', errors="ignore").read(), 128) 
        for _, example in zip(range(4), corruption_dataset):
            x, y = example
            print('x:', ''.join([corruption_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([corruption_dataset.itos[int(c)] for c in y]))
    else:
        raise ValueError("Unknown dataset type in command line args: {}"
                .format(args.dataset_type))

