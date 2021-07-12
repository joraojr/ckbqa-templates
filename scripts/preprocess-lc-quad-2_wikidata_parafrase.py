"""
Preprocessing script for LC-QUAD-2 data.
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
                    line = line.replace('\n', '')
                    line_split = line.split(" ")
                vocab |= set(line_split)
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')


def split_data(X, y, dst_dir, le):
    with open(os.path.join(dst_dir, 'id.txt'), 'w') as idfile, \
            open(os.path.join(dst_dir, 'input.txt'), 'w') as inputfile, \
            open(os.path.join(dst_dir, 'output.txt'), 'w') as outputfile, \
            open(os.path.join(dst_dir, 'output_group.txt'), 'w') as groupfile, \
            open(os.path.join(dst_dir, 'question_type.txt'), 'w') as question_typefile:
        y = y.tolist()

        print((X.shape, len(y)))
        for index in range(len(X)):
            question = str(X.iloc[index]["question"]).strip().replace('"', "") \
                .replace("\n", " ").replace("{", "").replace("}", "").replace("<", "").replace(">", "")
            question = question.strip()
            corrected_question = question[:-1] if question.endswith(("?", ".", "!")) else question
            idfile.write(str(X.iloc[index]["uid"]) + "\n")
            inputfile.write(corrected_question + "\n")
            outputfile.write(str(le.transform([y[index]])[0]) + "\n")
            groupfile.write(str(X.iloc[index]["template_id_dummy_group"]) + "\n")
            question_typefile.write(str(X.iloc[index]["template_id"]) + "\n")


def parse(dirpath, cp=''):
    dependency_parse(os.path.join(dirpath, 'input.txt'), cp=cp, tokenize=True)
    # constituency_parse(os.path.join(dirpath, 'input.txt'), cp=cp, tokenize=True)


if __name__ == '__main__':
    RANDOM_SEED = 42
    print('=' * 80)
    print('Preprocessing LC-QUAD-2 dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    lc_quad_dir = os.path.join(data_dir, 'lc-quad-2-wikidata-parafrase')
    lib_dir = os.path.join(base_dir, 'lib')
    train_dir = os.path.join(lc_quad_dir, 'train')
    test_dir = os.path.join(lc_quad_dir, 'test')
    valid_dir = os.path.join(lc_quad_dir, 'dev')
    make_dirs([train_dir, test_dir, valid_dir])
    make_dirs([os.path.join(lc_quad_dir, 'pth')])

    # java classpath for calling Stanford parser
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.9.1-models.jar')])
    print(classpath)

    # Load Data

    df_dummy = pd.read_json(os.path.join(lc_quad_dir, "DummyTemplatesWikidata.json"))
    print(df_dummy.template_id.value_counts())
    df_full = []

    template_id_dict = {
        2: 1,
        1: 0,
        "1.1": 11,
        "statement_property_1": 7,
        "statement_property_2": 6,
        3: 8,
        5: 9,
        "Count_2": 3,
        "Rank2": 5,
        "1.2": 12,
        "1": 0,
        "Count_1": 2,
        4: 10,
        "Rank1": 4
    }

    import math

    for index, row in df_dummy.iterrows():
        df_full.append({
            "uid": row["uid"],
            "question": row["question"],
            "Dummy_wikidata": row["Dummy_wikidata"],
            "template_id_dummy": row["Dummy_id_wikidata"],
            "template_id_dummy_group": math.floor(row["Dummy_id_wikidata"]),
            "template_id": template_id_dict[row["template_id"]],
        })
        df_full.append({
            "uid": str(row["uid"]) + "_paraphrased_question",
            "question": row["paraphrased_question"],
            "Dummy_wikidata": row["Dummy_wikidata"],
            "template_id_dummy": row["Dummy_id_wikidata"],
            "template_id_dummy_group": math.floor(row["Dummy_id_wikidata"]),
            "template_id": template_id_dict[row["template_id"]],
        })

    del df_dummy
    df_full = pd.DataFrame(df_full)

    le = LabelEncoder()
    le.fit(df_full.template_id_dummy.values)

    np.save(os.path.join(lc_quad_dir, "le_dummy.npy"), le.classes_)

    print(df_full.template_id_dummy_group.value_counts())

    df_full = df_full[df_full.question.notnull()]
    df_full = df_full[df_full.astype(str)['question'] != '[]']

    df_full = df_full[~df_full.question.str.lower().isin(["n/a", "na"])]
    df_full.drop_duplicates(inplace=True, subset=["question"])
    df_full = df_full[~df_full.question.str.contains("\(|\)|{|}|<|>|_|\.\.|\[|\]|not appl", regex=True)]
    df_full.drop(df_full[df_full.question.str.startswith(('"', ";", ":"))].index, inplace=True)
    # df_full[(df_full.question.str.split("?").str.len() >= 3) |
    #        (df_full.question.str.split("?").str[-1].str.strip().str.len() > 0)].to_csv("revisar.csv")
    #
    df_full.drop(df_full[(df_full.question.str.split("?").str.len() >= 3)].index, inplace=True)
    df_full.drop(df_full[(df_full.question.str.split("?").str.len() >= 2) &
                         (df_full.question.str.split("?").str[-1].str.strip().str.len() > 0)].index, inplace=True)

    print(len(df_full))
    print(len(df_full.template_id_dummy))
    print(len(df_full["Dummy_wikidata"]))

    print(df_full)

    df_full[(df_full.question.str.len() >= 105) | (df_full.question.str.len() <= 10)].to_csv("olhar2.csv")

    print(df_full.template_id_dummy_group.value_counts())
    df_full = df_full[(df_full.question.str.len() < 105) & (df_full.question.str.len() > 10)]
    print(df_full.template_id_dummy_group.value_counts())

    pd.DataFrame(df_full.question.str.len()).boxplot().figure.savefig("./boxplot.png")

    df_full.drop_duplicates(inplace=True, subset=["question"])

    df_full.to_json("./df_full.json", orient="records", default_handler=str, indent=3)
    df_full.to_csv("./df_full.csv")

    df_train, df_test = train_test_split(df_full, shuffle=True, stratify=df_full.template_id_dummy, test_size=0.1,
                                         random_state=RANDOM_SEED)

    df_train, df_valid = train_test_split(df_train, shuffle=True, stratify=df_train.template_id_dummy, test_size=0.1,
                                          random_state=RANDOM_SEED)

    print(df_train.template_id_dummy.value_counts())
    print(df_valid.template_id_dummy.value_counts())
    print(df_test.template_id_dummy.value_counts())

    df_train.to_csv("./df_train.csv")
    df_valid.to_csv("./df_dev.csv")
    df_test.to_csv("./df_test.csv")

    df_train.to_json("./df_train_novo.json", orient="records", default_handler=str)
    df_test.to_json("./df_test_novo.json", orient="records", default_handler=str)
    df_valid.to_json("./df_dev_novo.json", orient="records", default_handler=str)

    X_train = df_train.drop(columns=['template_id_dummy'])
    y_train = pd.Series(df_train['template_id_dummy'].tolist())

    X_test = df_test.drop(columns=['template_id_dummy'])
    y_test = pd.Series(df_test['template_id_dummy'].tolist())

    X_valid = df_valid.drop(columns=['template_id_dummy'])
    y_valid = pd.Series(df_valid['template_id_dummy'].tolist())

    split_data(X_train, y_train, train_dir, le)
    split_data(X_test, y_test, test_dir, le)
    split_data(X_valid, y_valid, valid_dir, le)

    # parse sentences
    parse(train_dir, cp=classpath)
    parse(test_dir, cp=classpath)
    parse(valid_dir, cp=classpath)

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

    build_vocab(
        glob.glob(os.path.join(lc_quad_dir, '*/output_group.txt')),
        os.path.join(lc_quad_dir, 'vocab_output_group.txt'))

    build_vocab(
        glob.glob(os.path.join(lc_quad_dir, '*/question_type.txt')),
        os.path.join(lc_quad_dir, 'vocab_question_type.txt'))
