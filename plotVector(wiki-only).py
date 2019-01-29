from parseValidFiles import *
import matplotlib
import os
matplotlib.use('Agg')
from gensim import models
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly.offline as offline

i = 2

HOUHAN_DIR =  './test{:d}/'.format(i)
SAMPLE_DIR = './train{:d}/'.format(i)
NUCC_DIR = './nucc/'
LOAD_MODEL = 'master(wiki-only).model'
encoding = 'utf-8'

model = models.Doc2Vec.load(LOAD_MODEL)

def create_vectors(dir):
    vectors = []
    num = 0
    corpus = list(get_all_files(dir))
    sentences = list(corpus_to_sentences(corpus, encoding))
    for root, dirs, files in os.walk(dir):
        num += len(files)
    for idx in range(num):
        vectors.append(model.infer_vector(sentences[idx].words))
    return vectors

houhan = create_vectors(HOUHAN_DIR)
daily = create_vectors(NUCC_DIR)
sample = create_vectors(SAMPLE_DIR)

num_h = len(houhan)
num_d = len(daily)
num_s = len(sample)
num_all = num_h + num_d + num_s
print(num_h, num_d, num_s)

all_vecs = houhan + daily + sample

tsne_model = TSNE(n_components=3, random_state=0, verbose=2).fit_transform(all_vecs)

trace1 = go.Scatter3d(
    x=tsne_model[0:num_h, 0],
    y=tsne_model[0:num_h, 1],
    z=tsne_model[0:num_h, 2],
    name='houhan',
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color = 'rgb(255, 0, 0)',
        colorscale = 'Portland',
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.9,
        size=2
    )
)

trace2 = go.Scatter3d(
    x=tsne_model[num_h:num_h + num_d, 0],
    y=tsne_model[num_h:num_h + num_d, 1],
    z=tsne_model[num_h:num_h + num_d, 2],
    name='daily',
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color = 'rgb(0, 255, 0)',
        colorscale = 'Portland',
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.9,
        size=2
    )
)

trace3 = go.Scatter3d(
    x=tsne_model[num_h + num_d:num_h + num_d + num_s, 0],
    y=tsne_model[num_h + num_d:num_h + num_d + num_s, 1],
    z=tsne_model[num_h + num_d:num_h + num_d + num_s, 2],
    name='sample',
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color = 'rgb(0, 0, 255)',
        colorscale = 'Portland',
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.9,
        size=2
    )
)

data=[trace1, trace2, trace3]
layout=go.Layout(height=1000, width=1500, title='t-SNE plot validation{:d}'.format(i), legend={"x":0, "y":0})
fig=go.Figure(data=data, layout=layout)
offline.plot(fig, filename='t-SNE validation{:d}.html'.format(i))