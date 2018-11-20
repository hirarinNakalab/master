import os
import sys
import MeCab
import collections
from gensim import models
from gensim.models.doc2vec import LabeledSentence


ENC_CONFIG = 'utf-8'
INPUT_DOC_DIR = './text/'


# 全てのファイルのリストを取得
def get_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

# ファイルから文章を返す
def read_document(path):
    with open(path, 'r', encoding=ENC_CONFIG, errors='ignore') as f:
        return f.read()

# 青空文庫ファイルから作品部分のみ抜き出す
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
        if line.startswith('＃'):
            line = line[1:]
        if line.startswith('％'):
            continue
        if line.startswith('F'):
            line = line[5:]
        if line.startswith('M'):
            line = line[5:]
        #print(line)
        valid_lines.append(line)
    
    return ''.join(valid_lines)

# 文章から単語に分解して返す
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

# ファイルから単語のリストを取得
def corpus_to_sentences(corpus):
    docs = [read_document(x) for x in corpus]
    for idx, (doc, name) in enumerate(zip(docs, corpus)):
        sys.stdout.write('\r前処理中 {} / {}'.format(idx, len(corpus)))
        yield split_into_words(doc, name)

# 学習
def train(sentences):
    model = models.Doc2Vec(vector_size=300, epochs=20, start_alpha=0.0015, end_alpha=0.0015, sample=1e-4, min_count=1, workers=4, dm=1)
#     model = models.Doc2Vec(vector_size=300, epochs=20, start=0.025, end=0.0001, sample=1e-5, min_count=1, workers=4, window=15, dm=0)
    model.build_vocab(sentences)    
    model.train(sentences, epochs=model.epochs, total_examples=model.corpus_count)
    
    return model

if __name__ == '__main__':
    wiki_corpus = list(get_all_files(INPUT_DOC_DIR))
    wiki_sentences = list(corpus_to_sentences(wiki_corpus))
    OUTPUT_MODEL = 'master(wiki-only).model'
    model = train(wiki_sentences)
    model.save(OUTPUT_MODEL)
#     for i in range(3):
#         TRAIN_HH_DIR = './train{:d}/'.format(i)
#         OUTPUT_MODEL = 'master(validation{:d}).model'.format(i)

#         houhan_corpus = list(get_all_files(TRAIN_HH_DIR))
#         houhan_sentences = list(corpus_to_sentences(houhan_corpus))
        
#         sentences = wiki_sentences + houhan_sentences
        
#         model = train(sentences)
#         model.save(OUTPUT_MODEL)