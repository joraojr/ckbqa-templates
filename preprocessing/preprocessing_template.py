"""
Template preprocessing script for use HTL
"""

import glob
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
                    line = line.replace('\n', '')
                    line_split = line.split(" ")
                vocab |= set(line_split)
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')


def split_data(X, y, dst_dir):
    with open(os.path.join(dst_dir, 'id.txt'), 'w') as idfile, \
            open(os.path.join(dst_dir, 'input.txt'), 'w') as inputfile, \
            open(os.path.join(dst_dir, 'output.txt'), 'w') as outputfile:
        # open(os.path.join(dst_dir, 'output_group.txt'), 'w') as groupfile, \
        # open(os.path.join(dst_dir, 'question_type.txt'),'w') as question_typefile:
        y = y.tolist()

        # TODO replace here with you dataset column name.
        for index in range(len(X)):
            corrected_question = str(X.iloc[index]["XXXX"]).strip()
            idfile.write(str(X.iloc[index]["XXXX"]) + "\n")
            inputfile.write(corrected_question + "\n")
            outputfile.write(str(y[index]) + "\n")


def parse(dirpath, cp=''):
    dependency_parse(os.path.join(dirpath, 'input.txt'), cp=cp, tokenize=True)
    # constituency_parse(os.path.join(dirpath, 'input.txt'), cp=cp, tokenize=True)


if __name__ == '__main__':
    """
    
    """

    print('=' * 80)
    print('Preprocessing XXX dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    dataset_dir = os.path.join(data_dir, 'lc-quad')
    lib_dir = os.path.join(base_dir, 'lib')
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    make_dirs([train_dir, test_dir])
    make_dirs([os.path.join(dataset_dir, 'pth')])

    # java classpath for calling Stanford parser
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.9.1-models.jar')])

    # Load Data
    df_train = pd.read_json(None)
    df_test = pd.read_json(None)
    df = pd.concat([df_train, df_test], ignore_index=True)

    X = None
    y = None

    # TODO Create your preprocessing based on your dataset

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    split_data(X_train, y_train, train_dir)
    split_data(X_test, y_test, test_dir)

    # parse sentences
    parse(train_dir, cp=classpath)
    parse(test_dir, cp=classpath)

    # Build Vocabulary for input
    build_vocab(
        glob.glob(os.path.join(data_dir, '**/*.toks'), recursive=True),
        os.path.join(dataset_dir, 'vocab_toks.txt'), lowercase=True)

    # Character Level
    build_vocab(
        glob.glob(os.path.join(dataset_dir, '*/*.toks')),
        os.path.join(dataset_dir, 'vocab_chars.txt'), lowercase=True, character_level=True)

    build_vocab(
        glob.glob(os.path.join(dataset_dir, '*/*.pos')),
        os.path.join(dataset_dir, 'vocab_pos.txt'))

    build_vocab(
        glob.glob(os.path.join(dataset_dir, '*/*.rels')),
        os.path.join(dataset_dir, 'vocab_rels.txt'))

    # Build Vocabulary for output
    build_vocab(
        glob.glob(os.path.join(dataset_dir, '*/output.txt')),
        os.path.join(dataset_dir, 'vocab_output.txt'))

    # TODO add other Vocabulary if needed
#    build_vocab(
#        glob.glob(os.path.join(dataset_dir, '*/output_group.txt')),
#        os.path.join(dataset_dir, 'vocab_output_group.txt'))
#
#    build_vocab(
#        glob.glob(os.path.join(dataset_dir, '*/question_type.txt')),
#        os.path.join(dataset_dir, 'vocab_question_type.txt'))
