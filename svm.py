import itertools
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from gensim import models
from parseValidFiles import *
import warnings
warnings.filterwarnings('ignore')

LOAD_MODEL = 'master(wiki-only).model'
model = models.Doc2Vec.load(LOAD_MODEL)
encoding = 'utf-8'

def create_vectors(dir):
    vectors = []
    num = 0
    corpus = list(get_all_files(dir))
    sentences = list(corpus_to_sentences(corpus, encoding))
    for root, dirs, files in os.walk(dir):
        num += len(files)
    for idx in range(num):
        vectors.append(model.infer_vector(sentences[idx].words))
    return np.array(vectors)

def create_X_Y(DIRS):
    x = np.concatenate([create_vectors(dir) for dir in DIRS], axis=0)
    y = []
    for i, dir in enumerate(DIRS):
        num = 0
        for root, dirs, files in os.walk(dir):
            num += len(files)
        y = y + [i for _ in range(num)]
    return x, np.array(y)

def plot_confusion_matrix(cm, classes, output_file, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_file)
    plt.clf()

class Variable:
    def __init__(self):
        self.BASE_DIR = './matsumoto/houhan_doc/'
        self.HIGH_DIR = self.BASE_DIR + 'high'
        self.MID_DIR =  self.BASE_DIR + 'middle'
        self.LOW_DIR =  self.BASE_DIR + 'low'
        self.labels = ['high', 'middle', 'low']
        self.DIRS = [self.HIGH_DIR, self.MID_DIR, self.LOW_DIR]


variable = Variable()

x, y = create_X_Y(variable.DIRS)

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

parameters = [
    {'kernel':['rbf'], 'gamma':np.linspace(0.001, 0.004, 50), 'C':np.linspace(10, 20, 50)},
    # {'kernel':['linear'], 'C': [1, 10, 100, 1000]}
]
gs = GridSearchCV(svm.SVC(class_weight="balanced"), param_grid=parameters, cv=5)

precision = []
recall = []
f1_sc = []
i = 0
for train, test in kfold.split(x, y):
    sm = SMOTE(random_state=42, kind='svm')
    x_res, y_res = sm.fit_resample(x[train], y[train])
    gs.fit(x_res, y_res)
    # gs.fit(x[train], y[train])
    print('Best score', gs.best_score_)
    print('Best model', gs.best_estimator_)

    pred = gs.predict(x[test])
    cm = confusion_matrix(y[test], pred)
    report = classification_report(y[test], pred)
    with open('report{:d}.txt'.format(i), 'w') as f:
        f.write(report)
    plot_confusion_matrix(cm, classes=variable.labels, output_file='{:d}.png'.format(i))
    i = i + 1