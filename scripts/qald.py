import os

import numpy as np
import torch
import json

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

from torch.autograd import Variable as Var
from treelstm import Vocab
from treelstm import LC_QUAD_Dataset
from treelstm import Constants

from main import generate_embeddings


def dependency_parse(filepath, cp='', tokenize=True):
    print('\nDependency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.parents')
    relpath = os.path.join(dirpath, filepre + '.rels')
    pospath = os.path.join(dirpath, filepre + '.pos')
    lenpath = os.path.join(dirpath, filepre + '.len')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    parser_lib = os.path.join(base_dir, "lib/DependencyParse.java")
    cmd = ('java -cp %s %s -tokpath %s -parentpath %s -relpath %s -pospath %s -lenpath %s %s < %s'
           % (cp, parser_lib, tokpath, parentpath, relpath, pospath, lenpath, tokenize_flag, filepath))
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    qald_dir = os.path.join(data_dir, 'qald-7-wikidata')
    lib_dir = os.path.join(base_dir, 'lib')

    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.9.1-models.jar')])

    data = json.load(open(os.path.join(qald_dir, 'qald-7-train-en-wikidata-dummy-ids.json')))

    questions = []
    sparql = []
    sparql_dummy = []
    id_dummy = []

    with open(os.path.join(qald_dir, 'input.txt'), 'w') as inputfile, open(os.path.join(qald_dir, 'id.txt'),
                                                                           'w') as idfile:
        for q in data['questions']:
            for question in q["question"]:
                if question["language"] == "en":
                    nlquestion = question["string"]
                    questions.append(nlquestion)
                    break
            sparql.append(q["query"]["sparql"])
            sparql_dummy.append(q["query"]["sparql_dummy"])
            id_dummy.append(q["query"]["id_dummy"])
            inputfile.write(nlquestion + "\n")
            idfile.write(str(q["id"]) + "\n")

    dependency_parse(os.path.join(qald_dir, 'input.txt'), cp=classpath)

    saved_model = torch.load(os.path.join(base_dir,
                                          'checkpoints-lcquad2-wikidata/with--attention--3,epoch=45,test_acc=0.7136363636363636.pt'))
    trainer = saved_model['trainer']

    vocab_rels = Vocab(filename=os.path.join(base_dir, 'data/lc-quad-2-wikidata/vocab_rels.txt'))
    vocab_pos = Vocab(filename=os.path.join(base_dir, 'data/lc-quad-2-wikidata/vocab_pos.txt'))
    vocab_toks = Vocab(filename=os.path.join(base_dir, 'data/lc-quad-2-wikidata/vocab_toks.txt'),
                       data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    vocab_output = trainer.vocabs['output']

    toks_emb = generate_embeddings(vocab_toks,
                                   os.path.join(base_dir, 'data/lc-quad-2-wikidata/pth/lc_quad_toks_embed.pth'))
    toks_embedding_model = nn.Embedding(vocab_toks.size(), 300)
    toks_embedding_model.state_dict()['weight'].copy_(toks_emb)
    trainer.embeddings['toks'] = toks_embedding_model
    trainer.vocabs['toks'] = vocab_toks

    train_dataset = LC_QUAD_Dataset(os.path.join(base_dir, 'data/qald-7-wikidata'), vocab_toks, vocab_pos, vocab_rels,
                                    0)

    json_object = []
    for index in range(len(train_dataset)):
        tree, toks_sent, pos_sent, rels_sent, label = train_dataset[index]
        toks_sent = Var(toks_sent)
        pos_sent = Var(pos_sent)
        rels_sent = Var(rels_sent)

        toks_emb = torch.unsqueeze(trainer.embeddings['toks'](toks_sent), 1)
        pos_emb = torch.unsqueeze(trainer.embeddings['pos'](pos_sent), 1)
        rels_emb = torch.unsqueeze(trainer.embeddings['rels'](rels_sent), 1)
        # chars_emb = trainer.get_char_vector(toks_sent)
        emb = torch.cat((toks_emb, pos_emb, rels_emb), 2)

        output = trainer.model.forward(tree, emb, training=False)
        _, pred = torch.topk(output[0].squeeze(0), 2)
        pred = pred.numpy()

        encoder = LabelEncoder()
        encoder.classes_ = np.load(
            '/Users/joraojr./Documents/Mestrado/ckbqa-templates/data/lc-quad-2-wikidata/le_dummy.npy')

        json_object.append({
            'question': questions[index],
            'sparql': sparql[index],
            'sparql_dummy': sparql_dummy[index],
            'id_dummy': id_dummy[index],
            'predictions': [int(vocab_output.getLabel(pred[0])), int(vocab_output.getLabel(pred[1]))],
            'labelencoder': str(encoder.inverse_transform([int(vocab_output.getLabel(pred[0]))])) + ',' + str(
                encoder.inverse_transform([int(vocab_output.getLabel(pred[1]))]))
            # 'actual': str(data.loc[index, 'template'])
        })

    with open(os.path.join(base_dir, 'data/qald-7-wikidata/qald_train_com_attent.json'), 'w') as outfile:
        json.dump(json_object, outfile)
