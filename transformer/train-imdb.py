import util

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext import vocab

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, gzip

# Used for converting between nats and bits.
LOG2E = math.log2(math.e)
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)
NUM_CLS = 2


def train(args):
    tbw = SummaryWriter(log_dir="runs")

    if args.final:
        train, test = datasets.IMDB.splits(TEXT, LABEL)
    else:
        train_data, _ = datasets.IMDB.splits(TEXT, LABEL)
        train, test = train_data.split(split_ratio=0.8)

    # -2 to make space for <unk> and <pad>.
    TEXT.build_vocab(train, max_size=args.vocab_size - 2)
    LABEL.build_vocab(train)

    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=args.batch_size, device=util.d()
    )

    print(f"- nr. of training examples {len(train_iter)}")
    print(
        f'- nr. of {"test" if args.final else "validation"} examples {len(test_iter)}'
    )

    if args.max_length < 0:
        mx = max([input.text[0].size(1) for input in train_iter])
        mx = mx * 2
        print(f"- maximum sequence length: {mx}")
    else:
        mx = args.max_length


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "-e",
        "--num-epochs",
        dest="num_epochs",
        help="Number of epochs.",
        default=80,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        help="The batch size.",
        default=64,
        type=int,
    )
    parser.add_argument(
        "-l",
        "--learn-rate",
        dest="lr",
        help="Learning rate.",
        default=0.0001,
        type=float,
    )
    parser.add_argument(
        "-T",
        "--tb_dir",
        dest="tb_dir",
        help="Tensorboard logging directory.",
        default="./runs",
    )
    parser.add_argument(
        "-f",
        "--final",
        dest="final",
        help="Whether to run on the real test set(if not included, the validation set is used.",
        action="store_true",
    )
    parser.add_argument(
        "--max-pool",
        dest="max_pool",
        help="Use max pooling in the final classification layer.",
        action="store_true",
    )
    parser.add_argument(
        "-E",
        "--embedding",
        dest="embedding_size",
        help="Size of the character embeddings.",
        default=128,
        type=int,
    )
    parser.add_argument(
        "-V",
        "--vocab-size",
        dest="vocab_size",
        help="Max sequence length. Longer ones are clipped.",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "-H",
        "--heads",
        dest="num_heads",
        help="Number of attention heads.",
        default=8,
        type=int,
    )
    parser.add_argument(
        "-d",
        "--dpeth",
        dest="depth",
        help="Depth of the network (nr of self-attention layers).",
        default=4,
        type=int,
    )
    parser.add_argument(
        "-r",
        "--random-seed",
        dest="seed",
        help="RNG seed. Negative for random.",
        default=1,
        type=int,
    )

    args = parser.parse_args()
    print("OPTIONS: ", args)

    train(args)
