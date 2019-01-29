from parseValidFiles import *
import matplotlib
import os
matplotlib.use('Agg')
from gensim import models
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly.offline as offline

i = 2

LOAD_MODEL = 'master(wiki-only).model'
NUCC_DIR = './nucc/'
encoding = 'utf-8'
HIGH_DIR = './test{:d}/high_test{:d}/'.format(i, i)
MIDDLE_DIR = './test{:d}/middle_test{:d}/'.format(i, i)
LOW_DIR = './test{:d}/low_test{:d}/'.format(i, i)
SAMPLE_DIR = './train{:d}/middle_train{:d}/'.format(i, i)


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

high = create_vectors(HIGH_DIR)
middle = create_vectors(MIDDLE_DIR)
low = create_vectors(LOW_DIR)
sample = create_vectors(SAMPLE_DIR)

num_h = len(high)
num_m = len(middle)
num_l = len(low)
num_s = len(sample)
num_all = num_h + num_m + num_l + num_s
print(num_h, num_m, num_l, num_s)

all_vecs = high + middle + low + sample

tsne_model = TSNE(n_components=3, random_state=0, verbose=2).fit_transform(all_vecs)

trace1 = go.Scatter3d(
    x=tsne_model[0:num_h, 0],
    y=tsne_model[0:num_h, 1],
    z=tsne_model[0:num_h, 2],
    name='high',
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
    x=tsne_model[num_h:num_h + num_m, 0],
    y=tsne_model[num_h:num_h + num_m, 1],
    z=tsne_model[num_h:num_h + num_m, 2],
    name='middle',
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
    x=tsne_model[num_h + num_m:num_h + num_m + num_l, 0],
    y=tsne_model[num_h + num_m:num_h + num_m + num_l, 1],
    z=tsne_model[num_h + num_m:num_h + num_m + num_l, 2],
    name='low',
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

trace4 = go.Scatter3d(
    x=tsne_model[num_h + num_m + num_l:num_all , 0],
    y=tsne_model[num_h + num_m + num_l:num_all , 1],
    z=tsne_model[num_h + num_m + num_l:num_all , 2],
    name='sample',
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color = 'rgb(192, 192, 192)',
        colorscale = 'Portland',
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.9,
        size=2
    )
)

data=[trace1, trace2, trace3, trace4]
layout=go.Layout(height=1000, width=1500, title='t-SNE plot validation{:d}'.format(i), legend={"x":0, "y":0})
fig=go.Figure(data=data, layout=layout)
offline.plot(fig, filename='t-SNE validation{:d}.html'.format(i))