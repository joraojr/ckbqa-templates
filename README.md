# A Hereditary Attentive Template-based Approach for Complex Knowledge Base Question Answering Systems

Knowledge Base Question Answering systems (KBQA) aim to find answers for natural language questions over a knowledge
base. Recent approaches have achieved promising results handling simple questions, but struggle to deal with constraint
and multiple hops questions. This repository presents a template matching approach for Complex KBQA systems (C-KBQA) using the combination of Semantic Parsing and Neural Networks techniques to classify natural language questions
into answer templates.

An attention mechanism was created to assist a Tree-LSTM in selecting the most important information. In the so-called
Hereditary Attention, each neural network cell inherits the attention from another neural network cell, in a bottom-up
way.

Furthermore, we presented the problems found in C-KBQA datasets and released a cleaned version of the LC-QuAD 2.0
dataset containing template answers, so-called LC-QuAD 2.1 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5508297.svg)](https://doi.org/10.5281/zenodo.5508297)


## Setup

Download Facebook FastText which is used as the embedding model:

```
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip # Download FastText
unzip wiki.en.zip
mv wiki.en.bin data/fasttext
rm wiki.en.zip
```

Download Stanford parser and tagger

```
wget -q -c https://nlp.stanford.edu/software/stanford-postagger-2018-02-27.zip
unzip -q stanford-postagger-2018-02-27.zip
mv stanford-postagger-2018-02-27/ lib/stanford-tagger
rm stanford-postagger-2018-02-27.zip

wget -q -c https://nlp.stanford.edu/software/stanford-parser-full-2018-02-27.zip
unzip -q stanford-parser-full-2018-02-27.zip
mv stanford-parser-full-2018-02-27/ lib/stanford-parser
rm stanford-parser-full-2018-02-27.zip
```

##Dataset

TBC

## How to use

```
$python main.py # Further configuration options can be found in config.py
```

---
**NOTE**

Part of this code was reused from this free
available [repository](https://github.com/ram-g-athreya/RNN-Question-Answering). We thank the authors for the first
version of the code.

---