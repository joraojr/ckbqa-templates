import os
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

from . import Constants
from .tree import Tree

import nltk


class LC_QUAD_Dataset(data.Dataset):
    def __init__(self, path, vocab_toks, vocab_pos, vocab_rels, num_classes):
        super(LC_QUAD_Dataset, self).__init__()
        self.vocab_toks = vocab_toks
        self.vocab_pos = vocab_pos
        self.vocab_rels = vocab_rels
        self.num_classes = num_classes
        self.toks_sentences = self.read_sentences(os.path.join(path, 'input.toks'), self.vocab_toks)
        self.pos_sentences = self.read_sentences(os.path.join(path, 'input.pos'), self.vocab_pos)
        self.rels_sentences = self.read_sentences(os.path.join(path, 'input.rels'), self.vocab_rels)
        self.trees = self.read_trees(os.path.join(path, 'input.parents'))

        if num_classes > 0:
            self.labels = self.read_labels(os.path.join(path, 'output.txt'))
        else:
            self.labels = torch.zeros(len(self.toks_sentences), dtype=torch.float)
        self.size = self.labels.size(0)
        # asd

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        tree = deepcopy(self.trees[index])
        toks_sent = deepcopy(self.toks_sentences[index])
        pos_sent = deepcopy(self.pos_sentences[index])
        rels_sent = deepcopy(self.rels_sentences[index])
        label = deepcopy(self.labels[index])
        return (tree, toks_sent, pos_sent, rels_sent, label)

    def read_sentences(self, filename, vocab):

        with open(filename, 'r') as f:
            sentences = []
            for line in tqdm(f.readlines()):
                indices = vocab.convertToIdx(line.split(), Constants.UNK_WORD)
                sentences.append(torch.tensor(indices, dtype=torch.long))
        return sentences

    def read_trees(self, filename):
        with open(filename, 'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_tree(self, line):
        parents = list(map(int, line.split()))
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_labels(self, filename):
        with open(filename, 'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
            labels = torch.tensor(labels, dtype=torch.float)
        return labels
