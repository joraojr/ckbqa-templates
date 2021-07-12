import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from . import utils
from . import Constants


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, num_classes, criterion, vocab_output, device):
        super(ChildSumTreeLSTM, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.num_classes = num_classes

        self.ix = nn.Linear(self.in_dim, self.mem_dim)
        self.ih = nn.Linear(self.mem_dim, self.mem_dim)

        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)

        self.ux = nn.Linear(self.in_dim, self.mem_dim)
        self.uh = nn.Linear(self.mem_dim, self.mem_dim)

        self.ox = nn.Linear(self.in_dim, self.mem_dim)
        self.oh = nn.Linear(self.mem_dim, self.mem_dim)

        self.criterion = criterion
        self.output_module = None
        self.vocab_output = vocab_output

    def set_output_module(self, output_module):
        self.output_module = output_module

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(torch.squeeze(child_h, 1), 0)

        i = torch.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = torch.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = torch.tanh(self.ux(inputs) + self.uh(child_h_sum))

        # add extra singleton dimension
        fx = torch.unsqueeze(self.fx(inputs), 1)
        f = torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = torch.sigmoid(f)

        fc = torch.squeeze(torch.mul(f, child_c), 1)

        c = torch.mul(i, u) + torch.sum(fc, 0)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, tree, inputs, training=False):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs, training)

        child_c, child_h = self.get_child_states(tree)
        child_c = child_c.to(self.device)
        child_h = child_h.to(self.device)
        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        output = self.output_module.forward(tree.state[1], training)

        return output

    def get_child_states(self, tree):
        """
        Get c and h of all children
        :param tree:
        :return: (tuple)
        child_c: (num_children, 1, mem_dim)
        child_h: (num_children, 1, mem_dim)
        """
        if tree.num_children == 0:
            child_c = Var(torch.zeros(1, 1, self.mem_dim))
            child_h = Var(torch.zeros(1, 1, self.mem_dim))
        else:
            child_c = Var(torch.Tensor(tree.num_children, 1, self.mem_dim))
            child_h = Var(torch.Tensor(tree.num_children, 1, self.mem_dim))
            for idx in range(tree.num_children):
                child_c[idx] = tree.children[idx].state[0]
                child_h[idx] = tree.children[idx].state[1]
        return child_c, child_h


class Classifier(nn.Module):
    def __init__(self, mem_dim, num_classes, dropout=False):
        super(Classifier, self).__init__()
        self.mem_dim = mem_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.l1 = nn.Linear(self.mem_dim, self.num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def set_dropout(self, dropout):
        self.dropout = dropout

    def forward(self, vec, training=False):
        if self.dropout:
            out = self.logsoftmax(self.l1(F.dropout(vec, training=training, p=0.2)))
        else:
            out = self.logsoftmax(self.l1(vec))
        return out


# putting the whole model together
class TreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, num_classes, criterion, vocab_output, device, dropout=False):
        super(TreeLSTM, self).__init__()
        self.tree_module = ChildSumTreeLSTM(in_dim, mem_dim, num_classes, criterion, vocab_output, device)
        self.tree_module.to(device)
        self.classifier = Classifier(mem_dim, num_classes, dropout)
        self.classifier.to(device)

        self.tree_module.set_output_module(self.classifier)

    def set_dropout(self, dropout):
        self.classifier.set_dropout(dropout)

    def forward(self, tree, inputs, training=False):
        output = self.tree_module(tree, inputs, training)
        return output


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        # self.chanel_in = in_dim
        # self.activation = activation
        # self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        # self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        # self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.query = nn.Linear(in_dim, in_dim)
        self.normalizer = 1 / np.sqrt(in_dim)

        self.softmax = nn.Softmax(dim=-1)  # last dim

    def forward(self, x):
        """
            inputs :
                x : children hidden states maps( B X N_children X MEM_DIM)
            returns :
                out : self attention value
                attention: B X MEM_DIM X MEM_DIM (tree node attention)
        """

        # m_batchsize, C, width, height = x.size()
        # proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        # proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        # energy = torch.bmm(proj_query, proj_key)  # transpose check
        # attention = self.softmax(energy)  # BX (N) X (N)
        # proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out = out.view(m_batchsize, C, width, height)
        # out = self.gamma * out + x

        # m_batchsize, num_child, mem_dim = x.size()
        proj_query = self.query(x).permute(0, 2, 1)
        proj_key = self.key(x)
        energy = torch.bmm(proj_query, proj_key).mul_(self.normalizer)  # transpose check
        attention = self.softmax(energy)
        proj_value = self.value(x)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        return out, attention


# module for childsumtreelstm
class AttChildSumTreeLSTM(nn.Module):

    def __init__(self, in_dim, mem_dim, num_classes, criterion, vocab_output, device):
        super(AttChildSumTreeLSTM, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.num_classes = num_classes

        self.attention = Self_Attn(self.mem_dim, "relu")

        self.ag = nn.Linear(self.mem_dim, self.mem_dim)

        self.ix = nn.Linear(self.in_dim, self.mem_dim)
        self.ih = nn.Linear(self.mem_dim, self.mem_dim)

        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)

        self.ux = nn.Linear(self.in_dim, self.mem_dim)
        self.uh = nn.Linear(self.mem_dim, self.mem_dim)

        self.ox = nn.Linear(self.in_dim, self.mem_dim)
        self.oh = nn.Linear(self.mem_dim, self.mem_dim)

        self.criterion = criterion
        self.output_module = None
        self.vocab_output = vocab_output

    def set_output_module(self, output_module):
        self.output_module = output_module

    def node_forward(self, inputs, child_c, child_h):

        child_h_sum, att = self.attention(child_h)
        child_h_sum = torch.sum(child_h_sum.squeeze(0), 0)

        i = torch.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = torch.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = torch.tanh(self.ux(inputs) + self.uh(child_h_sum))

        # add extra singleton dimension

        fx = torch.unsqueeze(self.fx(inputs), 0)
        f = torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = torch.sigmoid(f)

        fc = torch.squeeze(torch.mul(f, child_c), 0)
        c = torch.mul(i, u) + torch.sum(fc, 0)
        h = torch.mul(o, torch.tanh(c))
        return c, h, att

    def forward(self, tree, inputs, training=False):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs, training)

        child_c, child_h = self.get_child_states(tree)
        child_c = child_c.to(self.device)
        child_h = child_h.to(self.device)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)

        output = self.output_module.forward(tree.state[1], training)

        return output

    def get_child_states(self, tree):
        """
        Get c and h of all children
        :param tree:
        :return: (tuple)
        child_c: (1,num_children, mem_dim)
        child_h: (1, num_children, mem_dim)
        """
        if tree.num_children == 0:
            child_c = Var(torch.zeros(1, 4, self.mem_dim))
            child_h = Var(torch.zeros(1, 4, self.mem_dim))
        else:
            child_c = Var(torch.Tensor(1, tree.num_children, self.mem_dim))
            child_h = Var(torch.Tensor(1, tree.num_children, self.mem_dim))
            for idx in range(tree.num_children):
                child_c[0][idx] = tree.children[idx].state[0]
                child_h[0][idx] = tree.children[idx].state[1]
        return child_c, child_h


class AttTreeLSTM(nn.Module):

    def __init__(self, in_dim, mem_dim, num_classes, criterion, vocab_output, device, dropout=False):
        super(AttTreeLSTM, self).__init__()
        self.tree_module = AttChildSumTreeLSTM(in_dim, mem_dim, num_classes, criterion, vocab_output, device)
        self.tree_module.to(device)
        self.classifier = Classifier(mem_dim, num_classes, dropout)
        self.classifier.to(device)

        self.tree_module.set_output_module(self.classifier)

    def forward(self, tree, inputs, training=False):
        output = self.tree_module(tree, inputs, training)
        return output
