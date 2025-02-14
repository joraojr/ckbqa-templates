from __future__ import division
from __future__ import print_function

import os
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim

import csv

# IMPORT CONSTANTS
from treelstm import Constants
# NEURAL NETWORK MODULES/LAYERS
from treelstm import TreeLSTM, AttTreeLSTM
# DATA HANDLING CLASSES
from treelstm import Vocab
# DATASET CLASS FOR LC_QUAD DATASET
from treelstm import Dataset
# METRICS CLASS FOR EVALUATION
from treelstm import Metrics
# UTILITY FUNCTIONS
from treelstm import utils
# TRAIN AND TEST HELPER FUNCTIONS
from treelstm import Trainer
# CONFIG PARSER
from config import parse_args

from fasttext import load_model

EMBEDDING_DIM = 300


def generate_one_hot_vectors(vocab, file):
    emb_file = os.path.join(file)
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        emb = torch.zeros(vocab.size(), vocab.size(), dtype=torch.float)
        for word in vocab.labelToIdx.keys():
            word_index = vocab.getIndex(word)
            word_vector = torch.zeros(1, vocab.size())
            word_vector[0, word_index] = 1
            emb[word_index] = word_vector
            # emb[word_index] = torch.Tensor(vocab.size()).uniform_(-1, 1)
        torch.save(emb, emb_file)

    return emb


def generate_embeddings(vocab, file):
    emb_file = os.path.join(file)
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        print("Generating FastText Word Vectors")
        emb = torch.zeros(vocab.size(), EMBEDDING_DIM, dtype=torch.float)
        fasttext_model = load_model("data/fasttext/wiki.en.bin")

        for word in vocab.labelToIdx.keys():
            word_vector = fasttext_model.get_word_vector(word)
            if word_vector.all() != None and len(word_vector) == EMBEDDING_DIM:
                emb[vocab.getIndex(word)] = torch.Tensor(word_vector)
            else:
                emb[vocab.getIndex(word)] = torch.Tensor(EMBEDDING_DIM).uniform_(-1, 1)

        torch.save(emb, emb_file)
    return emb


# MAIN BLOCK
def main():
    global args
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname) + '.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()
    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')

    # get vocab object from vocab file previously written
    vocab_toks = Vocab(filename=os.path.join(args.data, 'vocab_toks.txt'),
                       data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    vocab_chars = Vocab(filename=os.path.join(args.data, 'vocab_chars.txt'))
    vocab_pos = Vocab(filename=os.path.join(args.data, 'vocab_pos.txt'))
    vocab_rels = Vocab(filename=os.path.join(args.data, 'vocab_rels.txt'))
    # vocab_question_type = Vocab(filename=os.path.join(args.data, 'vocab_question_type.txt'))

    if not args.use_group:
        vocab_output = Vocab(filename=os.path.join(args.data, 'vocab_output.txt'))
    else:
        vocab_output = Vocab(filename=os.path.join(args.data, 'vocab_output_group.txt'))
    # Set number of classes based on vocab_output
    args.num_classes = vocab_output.size()

    logger.debug('==> Dataset vocabulary toks size : %d ' % vocab_toks.size())
    logger.debug('==> Dataset vocabulary chars size : %d ' % vocab_chars.size())
    logger.debug('==> Dataset vocabulary pos size : %d ' % vocab_pos.size())
    logger.debug('==> Dataset vocabulary rels size : %d ' % vocab_rels.size())
    logger.debug('==> Dataset output vocabulary size : %d ' % vocab_output.size())

    # load LC_QUAD dataset splits
    if not args.use_parafrase:
        train_file = os.path.join(args.data, 'pth/lc_quad_train.pth')
    else:
        train_dir = os.path.join(args.data, 'train_parafrase/')
        train_file = os.path.join(args.data, 'pth/lc_quad_train_parafrase.pth')

    if args.use_group:
        train_file = train_file + "_group"
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = Dataset(train_dir, vocab_toks, vocab_pos, vocab_rels, args.num_classes, args.use_group)
        torch.save(train_dataset, train_file)
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))

    dev_file = os.path.join(args.data, 'pth/lc_quad_dev.pth')
    if args.use_group:
        dev_file = dev_file + "_group"
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = Dataset(dev_dir, vocab_toks, vocab_pos, vocab_rels, args.num_classes, args.use_group)
        torch.save(dev_dataset, dev_file)
    logger.debug('==> Size of test data    : %d ' % len(dev_dataset))

    test_file = os.path.join(args.data, 'pth/lc_quad_test.pth')
    if args.use_group:
        test_file = test_file + "_group"

    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = Dataset(test_dir, vocab_toks, vocab_pos, vocab_rels, args.num_classes, args.use_group)
        torch.save(test_dataset, test_file)
    logger.debug('==> Size of test data    : %d ' % len(test_dataset))

    criterion = nn.NLLLoss()

    input_dim = vocab_pos.size() + vocab_rels.size() + EMBEDDING_DIM  # + 1  # vocab_chars.size()

    if args.attention:
        model = AttTreeLSTM(
            input_dim,
            args.mem_dim,
            args.num_classes,
            criterion,
            vocab_output,
            device,
            dropout=True
        )
    else:
        model = TreeLSTM(
            input_dim,
            args.mem_dim,
            args.num_classes,
            criterion,
            vocab_output,
            device,
            dropout=True
        )

    toks_embedding_model = nn.Embedding(vocab_toks.size(), EMBEDDING_DIM)
    #    chars_embedding_model = nn.Embedding(vocab_chars.size(), vocab_chars.size())
    pos_embedding_model = nn.Embedding(vocab_pos.size(), vocab_pos.size())
    rels_embedding_model = nn.Embedding(vocab_rels.size(), vocab_rels.size())

    toks_emb = generate_embeddings(vocab_toks, os.path.join(args.data, 'pth/lc_quad_toks_embed.pth'))
    #    chars_emb = generate_one_hot_vectors(vocab_chars ,'pth/lc_quad_char_embed.pth')
    pos_emb = generate_one_hot_vectors(vocab_pos, os.path.join(args.data, 'pth/lc_quad_pos_embed.pth'))
    rels_emb = generate_one_hot_vectors(vocab_rels, os.path.join(args.data, 'pth/lc_quad_rels_embed.pth'))

    # plug these into embedding matrix inside model
    toks_embedding_model.state_dict()['weight'].copy_(toks_emb)
    #    chars_embedding_model.state_dict()['weight'].copy_(chars_emb)
    pos_embedding_model.state_dict()['weight'].copy_(pos_emb)
    rels_embedding_model.state_dict()['weight'].copy_(rels_emb)

    model.to(device), criterion.to(device)
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':

        optimizer = optim.Adagrad(
            [
                {'params': model.parameters(),
                 'lr': args.lr}
            ],
            lr=args.lr,
            weight_decay=args.wd
        )

    elif args.optim == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd
        )

    metrics = Metrics()
    #    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.05)

    # create trainer object for training and testing
    trainer = Trainer(args,
                      model,
                      {'toks': toks_embedding_model,
                       'pos': pos_embedding_model, 'rels': rels_embedding_model},
                      #                       'chars': chars_embedding_model},
                      {'toks': vocab_toks, 'chars': vocab_chars, 'output': vocab_output},
                      criterion,
                      optimizer
                      )
    file_name = "{}/expname={},input_dim={},mem_dim={},lr={},emblr={},wd={},epochs={}".format(args.analysis,
                                                                                              args.expname, input_dim,
                                                                                              args.mem_dim, args.lr,
                                                                                              args.emblr, args.wd,
                                                                                              args.epochs)
    # if True:
    #     saved_model = torch.load(os.path.join(args.save,
    #                                           'HDT_group_parafrase,epoch=34,test_acc=0.9387112956494132,dev_acc=0.909778171531005.pt'))
    #     trainer = saved_model['trainer']
    #     #scheduler = saved_model['scheduler']

    for epoch in range(args.epochs):
        print("\n" * 5)

        # Train Model
        trainer.train(train_dataset)

        # Test Model on Training Dataset
        # train_loss, train_pred = trainer.test(train_dataset)
        # train_acc = metrics.accuracy_score(train_pred, train_dataset.labels, vocab_output)

        # print('==> Train loss   : %f \t' % train_loss, end="")
        # print('Epoch ', str(epoch + 1), 'train percentage ', train_acc)
        # write_analysis_file(file_name, epoch, train_pred, train_dataset.labels, "train_acc", train_acc, train_loss,
        #                    vocab_output)

        # Dev Model on Testing Dataset
        dev_loss, dev_pred = trainer.test(dev_dataset)
        dev_acc = (metrics.accuracy_score(dev_pred, dev_dataset.labels, vocab_output),
                   metrics.balanced_accuracy(dev_pred, dev_dataset.labels, vocab_output)
                   )

        print('==> Dev loss   : %f \t' % dev_loss, end="")
        print('Epoch ', str(epoch + 1), 'dev percentage ', dev_acc)
        write_analysis_file(file_name, epoch, dev_pred, dev_dataset.labels, "dev_acc", dev_acc, dev_loss, vocab_output)

        # Test Model on Testing Dataset
        test_loss, test_pred = trainer.test(test_dataset)
        test_acc = (metrics.accuracy_score(test_pred, test_dataset.labels, vocab_output),
                    metrics.balanced_accuracy(test_pred, test_dataset.labels, vocab_output))

        print('==> Test loss   : %f \t' % test_loss, end="")
        print('Epoch ', str(epoch + 1), 'test percentage ', test_acc)
        write_analysis_file(file_name, epoch, test_pred, test_dataset.labels, "test_acc", test_acc, test_loss,
                            vocab_output)

        checkpoint_filename = '%s.pt' % os.path.join(args.save,
                                                     args.expname + ',epoch={},test_acc={},dev_acc={}'.format(epoch + 1,
                                                                                                              test_acc,
                                                                                                              dev_acc))
        checkpoint = {'trainer': trainer, 'dev_accuracy': dev_acc,
                      'test_accuracy': test_acc}  # , 'scheduler': scheduler}
        torch.save(checkpoint, checkpoint_filename)


#        scheduler.step()


def write_analysis_file(file_name, epoch, predictions, labels, accuracy_label, accuracy, loss, vocab_output):
    with open(file_name + ",current_epoch={},{}={},loss={}.csv".format(epoch + 1, accuracy_label, accuracy, loss),
              "w") as csv_file:
        writer = csv.writer(csv_file)
        preds = [vocab_output.getLabel(int(pred)) for pred in predictions]
        labels = labels.int().numpy()
        writer.writerows(zip(preds, labels))


if __name__ == "__main__":
    main()
