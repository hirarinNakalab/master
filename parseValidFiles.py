import os
import sys
import MeCab
import collections
from gensim import models
from gensim.models.doc2vec import LabeledSentence

INPUT_NUCC_DIR = './nucc/'
encoding = 'utf-8'

def get_all_files(in_dir):
    for root, dirs, files in os.walk(in_dir):
        for file in files:
            yield os.path.join(root, file)
            
def read_document(path, encoding):
    with open(path, 'r', encoding=encoding, errors='ignore') as f:
        return f.read()
    
def corpus_to_sentences(corpus, encoding):
    docs = [read_document(x, encoding) for x in corpus]
    for idx, (doc, name) in enumerate(zip(docs, corpus)):
        yield split_into_words(doc, name)
        
def trim_doc(doc):
    lines = doc.splitlines()
    valid_lines = []
    for line in lines:
        if line == '':
            continue
        if line.startswith('<doc') or line.startswith('</doc'):
            continue
        if "colspan" in line or "|||||" in line:
                continue
        if '＠'in line:
            continue
        if line.startswith('％'):
            continue
        if line.startswith('F'):
            line = line[5:]
        if line.startswith('＃'):
            line = line[1:]
        if line.startswith('M'):
            line = line[5:] 
        valid_lines.append(line)
    
    return ''.join(valid_lines)

def split_into_words(doc, name=''):
    mecab = MeCab.Tagger("-Ochasen")
    valid_doc = trim_doc(doc)
    lines = mecab.parse(valid_doc).splitlines()
    
    words = []
    for line in lines:
        chunks = line.split('\t')
        if len(chunks) > 3 and (chunks[3].startswith('動詞') or chunks[3].startswith('形容詞') or (chunks[3].startswith('名詞') and not chunks[3].startswith('名詞-数'))):
            words.append(chunks[0])
    return LabeledSentence(words=words, tags=[name])

if __name__ == "__main__":
    INPUT_VALI_DIR = './test{:d}/'.format(i)
    corpus = list(get_all_files(INPUT_VALI_DIR))# + list(get_all_files(INPUT_NUCC_DIR)) 
    vali_sentences = list(corpus_to_sentences(corpus, encoding))
    
