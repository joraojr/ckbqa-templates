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
            outputfile.write(str(le.transform([y[index]])) + "\n")


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
            outputfile.write(str(le.transform([y[index]])) + "\n")


def parse(dirpath, cp=''):
    dependency_parse(os.path.join(dirpath, 'input.txt'), cp=cp, tokenize=True)
    # constituency_parse(os.path.join(dirpath, 'input.txt'), cp=cp, tokenize=True)


if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing LC-QUAD-2 dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    lc_quad_dir = os.path.join(data_dir, 'lc-quad-2-dummy-2fases')
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

    df_dummy = pd.read_json(os.path.join(lc_quad_dir, "lcquald2_dummys_8-5-21.json"))
    df_train = pd.read_json(os.path.join(lc_quad_dir, "train.json"))
    df_test = pd.read_json(os.path.join(lc_quad_dir, "test.json"))

    import math

    for index, row in df_train.iterrows():
        dummy_db = df_dummy[df_dummy["uid"] == row["uid"]]
        df_train.loc[index, "Dummy_dbpedia18"] = dummy_db["Dummy_dbpedia18"].values[0].strip()
        df_train.loc[index, "template_id_dummy"] = dummy_db["Dummy_id_dbpedia18"].values[0]

    for index, row in df_test.iterrows():
        dummy_db = df_dummy[df_dummy["uid"] == row["uid"]]
        df_test.loc[index, "Dummy_dbpedia18"] = dummy_db["Dummy_dbpedia18"].values[0].strip()
        df_test.loc[index, "template_id_dummy"] = dummy_db["Dummy_id_dbpedia18"].values[0]

    df_train = df_train[df_train.question.notnull()]
    df_train = df_train[~df_train.question.isin(["n/a", "na"])]
    df_train.drop_duplicates(inplace=True, subset=["question", "template_id_dummy"])

    df_test = df_test[df_test.question.notnull()]
    df_test = df_test[~df_test.question.isin(["n/a", "na"])]
    df_test.drop_duplicates(inplace=True, subset=["question", "template_id_dummy"])

    df_train.drop(df_train[df_train.question.str.contains("{|}|<|>", regex=True)].index, inplace=True)
    print(df_train[df_train.question.str.contains("{|}|<|>", regex=True)].template_id_dummy.value_counts())

    df_test.drop(df_test[df_test.question.str.contains("{|}|<|>", regex=True)].index, inplace=True)
    print(df_test[df_test.question.str.contains("{|}|<|>", regex=True)].template_id_dummy.value_counts())

    df_train.drop(df_train[df_train.question.str.startswith('"')].index, inplace=True)
    print(df_train[df_train.question.str.startswith('"')].template_id_dummy.value_counts())

    df_test.drop(df_test[df_test.question.str.startswith('"')].index, inplace=True)
    print(df_test[df_test.question.str.startswith('"')].template_id_dummy.value_counts())

    #    df = pd.concat([df_train, df_test], ignore_index=True)
    #    df = df[df.question.notnull()]

    desired_templates = pd.read_csv(os.path.join(lc_quad_dir, "new_templates_dummy.csv"))
    print(desired_templates["Dummy Template dbpedia18"])


    def to_template_index(row):
        print(row['Dummy_dbpedia18'])
        x = desired_templates.loc[
            desired_templates["Dummy Template dbpedia18"] == row['Dummy_dbpedia18'], "id"].values[0]
        return x


    #    df_train['template_id_dummy'] = df_train.apply(
    #        lambda row: to_template_index(row),
    #        axis=1
    #    )

    print(len(df_train))
    print(len(df_train.template_id_dummy))
    print(len(df_train["Dummy_dbpedia18"]))

    print(df_train)

    X_train = df_train.drop(columns=['template_id_dummy'])
    y_train = pd.Series(df_train['template_id_dummy'].tolist())
    print("Train: {}".format(df_train['template_id_dummy'].value_counts()))
    # "24123 x 6027"

    # df_test = df_test[df_test["template"].isin(desired_templates["template"])]
    #    df_test['template_id_dummy'] = df_test.apply(
    #        lambda row: to_template_index(row),
    #        axis=1
    #    )
    print(df_test)

    X_test = df_test.drop(columns=['template_id_dummy'])
    y_test = pd.Series(df_test['template_id_dummy'].tolist())

    print("Test: {}".format(df_test['template_id_dummy'].value_counts()))

    le = LabelEncoder()
    le.fit(df_train.template_id_dummy.values)

    np.save(os.path.join(lc_quad_dir, "le_dummy.npy"), le.classes_)

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
