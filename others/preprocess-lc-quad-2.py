"""
Preprocessing script for LC-QUAD data.
"""

import glob
import json
import os

import pandas as pd
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


def split_data(X, y, dst_dir):
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
            outputfile.write(str(y[index]) + "\n")
            if X.iloc[index]["paraphrased_question"]:
                corrected_question = str(X.iloc[index]["paraphrased_question"]).strip().replace("&nbsp", " ") \
                    .replace("\n", " ").replace("{", "").replace("}", "").replace("<", "").replace(">", "")
                idfile.write(str(X.iloc[index]["uid"]) + "_paraphrased_question" + "\n")
                inputfile.write(corrected_question + "\n")
                outputfile.write(str(y[index]) + "\n")

def split_data_test(X, y, dst_dir):
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
            outputfile.write(str(y[index]) + "\n")


def parse(dirpath, cp=''):
    dependency_parse(os.path.join(dirpath, 'input.txt'), cp=cp, tokenize=True)
    # constituency_parse(os.path.join(dirpath, 'input.txt'), cp=cp, tokenize=True)


if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing LC-QUAD-2 dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    lc_quad_dir = os.path.join(data_dir, 'lc-quad-2')
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

    # Load Data

    df_train = pd.read_json(os.path.join(lc_quad_dir, "train.json"))
    print("len train before: {}".format(len(df_train)))
    df_train = df_train[df_train.question.notnull()]
    print("len train after: {}".format(len(df_train)))
    print("len train with paraphrased : {}".format(len(df_train[df_train.paraphrased_question.notnull()])))

    df_test = pd.read_json(os.path.join(lc_quad_dir, "test.json"))
    print("len test before: {}".format(len(df_test)))
    df_test = df_test[df_test.question.notnull()]
    print("len test after : {}".format(len(df_test)))
    print("len test with paraphrased : {}".format(len(df_test.paraphrased_question.notnull())))

    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df[df.question.notnull()]

    desired_templates = df['template'].value_counts() >= 50
    desired_templates = desired_templates.index[desired_templates == True].tolist()

    print("Qnt of template > 50 = {}".format(len(desired_templates)))

    # TODO trocar isso aqui pra número value->template
    desired_templates = pd.DataFrame(desired_templates, columns=["template"]).reset_index()

    print(df['template'].value_counts())
    desired_templates.to_csv(os.path.join(lc_quad_dir, "template_id.csv"))

    df_train = df_train[df_train["template"].isin(desired_templates["template"])]
    df_train['template_id'] = df_train.apply(
        lambda row: desired_templates.loc[desired_templates["template"] == row['template'], ["index"]].index.item(),
        axis=1
    )
    X_train = df_train.loc[:, df.columns != 'template_id']
    y_train = pd.Series(df_train['template_id'].tolist())
    print("Train: {}".format(df_train['template_id'].value_counts()))
    #"24123 x 6027"
    df_test = df_test[df_test["template"].isin(desired_templates["template"])]
    df_test['template_id'] = df_test.apply(
        lambda row: desired_templates.loc[desired_templates["template"] == row['template'], ["index"]].index.item(),
        axis=1
    )
    X_test = df_test.loc[:, df.columns != 'template_id']
    y_test = pd.Series(df_test['template_id'].tolist())

    print("Test: {}".format(df_test['template_id'].value_counts()))

    split_data(X_train, y_train, train_dir)
    split_data_test(X_test, y_test, test_dir)

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
