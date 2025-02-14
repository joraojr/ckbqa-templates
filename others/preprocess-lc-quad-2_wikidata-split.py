"""
Preprocessing script for LC-QUAD data.
"""

import glob
import json
import pickle
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


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
    cmd = ('java -cp %s lib/DependencyParse.java -tokpath %s -parentpath %s -relpath %s -pospath %s -lenpath %s %s < %s'
           % (cp, tokpath, parentpath, relpath, pospath, lenpath, tokenize_flag, filepath))
    print(cmd)
    os.system(cmd)


def constituency_parse(filepath, cp='', tokenize=True):
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.cparents')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s lib/ConstituencyParse.java -tokpath %s -parentpath %s %s < %s'
           % (cp, tokpath, parentpath, tokenize_flag, filepath))
    os.system(cmd)


def build_vocab(filepaths, dst_path, lowercase=False, character_level=False):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()

                if character_level is True:
                    line = line.replace('\r', '').replace('\t', '')
                    line_split = [c for c in line]
                else:
                    line_split = line.split()
                vocab |= set(line_split)
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')


# def split(filepath, dst_dir):
#     with open(filepath) as datafile, \
#             open(os.path.join(dst_dir, 'id.txt'), 'w') as idfile, \
#             open(os.path.join(dst_dir, 'input.txt'), 'w') as inputfile, \
#             open(os.path.join(dst_dir, 'output.txt'), 'w') as outputfile:
#         data = json.load(datafile)
#         for datum in data:
#             idfile.write(datum["uid"] + "\n")
#             inputfile.write(datum["question"] + "\n")
#             outputfile.write(str(datum["template_id"]) + "\n")


def split_data(X, y, dst_dir, le):
    with open(os.path.join(dst_dir, 'id.txt'), 'w') as idfile, \
            open(os.path.join(dst_dir, 'input.txt'), 'w') as inputfile, \
            open(os.path.join(dst_dir, 'output.txt'), 'w') as outputfile:
        y = y.tolist()

        print((X.shape, len(y)))
        for index in range(len(X)):
            corrected_question = str(X.iloc[index]["question"]).strip().replace("&nbsp", " ") \
                .replace("\n", " ").replace("{", "").replace("}", "").replace("<", "").replace(">", "")
            idfile.write(str(X.iloc[index]["uid"]) + "\n")
            inputfile.write(corrected_question + "\n")
            outputfile.write(str(le.transform([y[index]])[0]) + "\n")


#            if X.iloc[index]["paraphrased_question"]:
#                corrected_question = str(X.iloc[index]["paraphrased_question"]).strip().replace("&nbsp", " ") \
#                    .replace("\n", " ").replace("{", "").replace("}", "").replace("<", "").replace(">", "")
#                idfile.write(str(X.iloc[index]["uid"]) + "_paraphrased_question" + "\n")
#                inputfile.write(corrected_question + "\n")
#                outputfile.write(str(y[index]) + "\n")


def split_data_test(X, y, dst_dir, le):
    with open(os.path.join(dst_dir, 'id.txt'), 'w') as idfile, \
            open(os.path.join(dst_dir, 'input.txt'), 'w') as inputfile, \
            open(os.path.join(dst_dir, 'output.txt'), 'w') as outputfile:
        y = y.tolist()

        print((X.shape, len(y)))
        for index in range(len(X)):
            corrected_question = str(X.iloc[index]["question"]).strip().replace("&nbsp", " ") \
                .replace("\n", " ").replace("{", "").replace("}", "").replace("<", "").replace(">", "")
            idfile.write(str(X.iloc[index]["uid"]) + "\n")
            inputfile.write(corrected_question + "\n")
            outputfile.write(str(le.transform([y[index]])[0]) + "\n")


def parse(dirpath, cp=''):
    dependency_parse(os.path.join(dirpath, 'input.txt'), cp=cp, tokenize=True)
    # constituency_parse(os.path.join(dirpath, 'input.txt'), cp=cp, tokenize=True)


if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing LC-QUAD-2 dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    lc_quad_dir = os.path.join(data_dir, 'lc-quad-2-wikidata-2fases-split')
    lib_dir = os.path.join(base_dir, 'lib')
    train_dir = os.path.join(lc_quad_dir, 'train')
    test_dir = os.path.join(lc_quad_dir, 'test')
    make_dirs([train_dir, test_dir])
    make_dirs([os.path.join(lc_quad_dir, 'pth')])

    # java classpath for calling Stanford parser
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.9.1-models.jar')])
    print(classpath)

    # Load Data
    import math

    df_dummy = pd.read_json(os.path.join(lc_quad_dir, "DummyTemplatesWikidata.json"))

    # df_dummy.rename(columns={"Dummy_id_wikidata": "template_id_dummy"})
    df_dummy = df_dummy[df_dummy.question.notnull()]
    df_dummy = df_dummy[~df_dummy.question.isin(["n/a", "na"])]
    df_dummy["template_id_dummy"] = np.floor(df_dummy["Dummy_id_wikidata"])
    # df_dummy["template_id_dummy"] = df_dummy["Dummy_id_wikidata"]
    df_dummy.drop_duplicates(inplace=True, subset=["question"])

    df_dummy.drop(df_dummy[df_dummy.question.str.contains("{|}|<|>", regex=True)].index, inplace=True)
    print(df_dummy[df_dummy.question.str.contains("{|}|<|>", regex=True)].template_id_dummy.value_counts())

    df_dummy.drop(df_dummy[df_dummy.question.str.startswith('"')].index, inplace=True)
    print(df_dummy[df_dummy.question.str.startswith('"')].template_id_dummy.value_counts())

    # df_dummy.drop_duplicates(inplace=True, subset=["question"])

    print(len(df_dummy))
    print(len(df_dummy.template_id_dummy))
    print(len(df_dummy["Dummy_wikidata"]))

    print(df_dummy)

    le = LabelEncoder()
    le.fit(df_dummy.template_id_dummy.values)

    np.save(os.path.join(lc_quad_dir, "le_dummy.npy"), le.classes_)

    X = df_dummy.drop(columns=['template_id_dummy'])
    Y = pd.Series(df_dummy['template_id_dummy'].tolist())

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.1, random_state=42)

    split_data(X_train, y_train, train_dir, le)
    split_data_test(X_test, y_test, test_dir, le)

    # parse sentences
    parse(train_dir, cp=classpath)
    parse(test_dir, cp=classpath)

    # Build Vocabulary for input
    build_vocab(
        glob.glob(os.path.join(data_dir, '**/*.toks'), recursive=True),
        os.path.join(lc_quad_dir, 'vocab_toks.txt'), lowercase=True)

    # Character Level
    build_vocab(
        glob.glob(os.path.join(lc_quad_dir, '*/*.toks')),
        os.path.join(lc_quad_dir, 'vocab_chars.txt'), lowercase=True, character_level=True)

    build_vocab(
        glob.glob(os.path.join(lc_quad_dir, '*/*.pos')),
        os.path.join(lc_quad_dir, 'vocab_pos.txt'))

    build_vocab(
        glob.glob(os.path.join(lc_quad_dir, '*/*.rels')),
        os.path.join(lc_quad_dir, 'vocab_rels.txt'))

    # Build Vocabulary for output
    build_vocab(
        glob.glob(os.path.join(lc_quad_dir, '*/output.txt')),
        os.path.join(lc_quad_dir, 'vocab_output.txt'))
