import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import random
import argparse
random.seed(0)

import dataset
import model
import trainer
import utils

# 111
# python src/run.py finetune vanilla wiki.txt --writing_params_path vanilla.model.params --finetune_corpus_path birth_places_train.tsv
# python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path birth_dev.tsv --outputs_path vanilla.nopretrain.dev.predictions
# python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path birth_test_inputs.tsv --outputs_path vanilla.nopretrain.test.predictions

# 222
# python src/run.py pretrain vanilla wiki.txt --writing_params_path vanilla.pretrain.params
# python src/run.py finetune vanilla wiki.txt --reading_params_path vanilla.pretrain.params --writing_params_path vanilla.finetune.params --finetune_corpus_path birth_places_train.tsv
# python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.finetune.params --eval_corpus_path birth_dev.tsv --outputs_path vanilla.pretrain.dev.predictions
# python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.finetune.params --eval_corpus_path birth_test_inputs.tsv --outputs_path vanilla.pretrain.test. predictions

# 333
# python src/run.py pretrain synthesizer wiki.txt --writing_params_path synthesizer.pretrain.params
argp = argparse.ArgumentParser()
argp.add_argument('function',
    help="Whether to pretrain, finetune or evaluate a model",
    choices=["pretrain", "finetune", "evaluate"])
argp.add_argument('variant',
    help="Which variant of the model to run ('vanilla' or 'synthesizer')",
    choices=["vanilla", "synthesizer"])
argp.add_argument('pretrain_corpus_path',
    help="Path of the corpus to pretrain on", default=None)
argp.add_argument('--reading_params_path',
    help="If specified, path of the model to load before finetuning/evaluation",
    default=None)
argp.add_argument('--writing_params_path',
    help="Path to save the model after pretraining/finetuning", default=None)
argp.add_argument('--finetune_corpus_path',
    help="Path of the corpus to finetune on", default=None)
argp.add_argument('--eval_corpus_path',
    help="Path of the corpus to evaluate on", default=None)
argp.add_argument('--outputs_path', default=None)
args = argp.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Keep the block size 128
# Why is the pretraining corpus always required (even if we're not pretraining?)
# It's because we're using it as a hack to always have the same vocabulary
# (that is, the same mapping from character to integer, and we build the 
# vocab from the pretraining corpus.)
block_size = 128
text = open(args.pretrain_corpus_path, errors='ignore').read()
pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)

# We don't suggest you change these hyperparameters, as they're known to work.
# use them for both the vanilla and the synthesizer models
mconf = model.GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
    n_layer=4, n_head=8, n_embd=256)

"""
Don't change above here; write your code below
"""

if args.variant == 'vanilla':
    # TODO [part c]: Make some model here
    model = model.GPT(mconf)
    model = model.to(device)
elif args.variant == 'synthesizer':
    # TODO [part g]: Make some other model here
    mconf = model.GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
    synthesizer=True, n_layer=4, n_head=8, n_embd=256)    
    
    model = model.GPT(mconf)
    model = model.to(device) 

# From here on, your code should be identical independent of which
# variant (vanilla or synthesizer) has been chosen.

if args.function == 'pretrain':
    assert args.pretrain_corpus_path is not None
    assert args.writing_params_path is not None
    # TODO [part f]:
    # - Given:
    #     1. A corpus specified in args.pretrain_corpus_path
    #     2. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. Pretrain the model on this corpus
    #     2. Save the resulting model in args.writing_params_path
    # - Make sure to use the following hyperparameters for pretraining:
    #     max_epochs=650
    #     batch_size=128
    #     learning_rate=6e-3
    #     lr_decay=True
    #     warmup_tokens=512*20
    #     final_tokens=200*len(pretrain_dataset)*block_size
    #     num_workers=4
    # 定义config
    tconf = trainer.TrainerConfig(max_epochs=650, batch_size=128, learning_rate=6e-3,
                                  lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(pretrain_dataset)*block_size,
                                  num_workers=0, ckpt_path=args.writing_params_path)
    trainer = trainer.Trainer(model, pretrain_dataset, None, tconf)
    trainer.train()

elif args.function == 'finetune':
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None
    # TODO [part c] [part f]:
    # - Given:
    #     1. A finetuning corpus specified in args.finetune_corpus_path
    #     2. A path args.reading_params_path containing pretrained model
    #         parameters, or None if finetuning without a pretrained model
    #     3. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. If args.reading_params_path is specified, load these parameters
    #         into the model
    #     2. Finetune the model on this corpus
    #     3. Save the resulting model in args.writing_params_path
    # - Make sure to use the following hyperparameters:
    #     Hyperparameters for finetuning WITHOUT a pretrained model:
    #         max_epochs=75
    #         batch_size=256
    #         learning_rate=6e-4
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #     Hyperparameters for finetuning WITH a pretrained model:
    #         max_epochs=10
    #         batch_size=256
    #         learning_rate=6e-4
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    text = open(args.finetune_corpus_path, 'r', encoding='utf-8').read()
    finetune_dataset = dataset.NameDataset(pretrain_dataset, text)

    # finetuning WITH a pretrained model
    if args.reading_params_path:
        # initialize a trainer instance and kick off finetuning
        # 设置了num_workers=0避免runtimeerror，也可以把它写在__main__里面
        tconf = trainer.TrainerConfig(max_epochs=10, batch_size=256, learning_rate=6e-4,
                                      lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(pretrain_dataset)*block_size,
                                      num_workers=0 if torch.cuda.is_available() else 0, ckpt_path=args.writing_params_path)

        model.load_state_dict(torch.load(args.reading_params_path))
        model.eval()

        trainer = trainer.Trainer(model, finetune_dataset, None, tconf)

    # finetuning WITHOUT a pretrained model
    # text覆盖了原本的wiki.txt, 没有pretrain, 模型只在birth_places_train.tsv训练
    else:
        # initialize a trainer instance and kick off finetuning
        # 设置了num_workers=0避免runtimeerror，也可以把它写在__main__里面
        tconf = trainer.TrainerConfig(max_epochs=75, batch_size=256, learning_rate=6e-4,
                                      lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(pretrain_dataset)*block_size,
                                      num_workers=0 if torch.cuda.is_available() else 0, ckpt_path=args.writing_params_path)
        trainer = trainer.Trainer(model, finetune_dataset, None, tconf)

    # finetune
    trainer.train()
    # raise NotImplementedError


elif args.function == 'evaluate':
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    model.load_state_dict(torch.load(args.reading_params_path))
    correct = 0
    total = 0
    with open(args.outputs_path, 'w') as fout:
        predictions = []
        for line in tqdm(open(args.eval_corpus_path)):
            x = line.split('\t')[0]
            x = x + '⁇'
            x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None,...].to(device)
            pred = utils.sample(model, x, 32, sample=False)[0]
            completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
            pred = completion.split('⁇')[1]
            predictions.append(pred)
            fout.write(pred + '\n')
        total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
        print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
    else:
        print('Predictions written to {}; no targets provided'
                .format(args.outputs_path))

