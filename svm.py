import matplotlib.pyplot as plt
import numpy as np
from gensim import models
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
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

class Variable:
    def __init__(self):
        self.BASE_DIR = './matsumoto/houhan_doc/'
        self.HIGH_DIR = self.BASE_DIR + 'high'
        self.MID_DIR =  self.BASE_DIR + 'middle'
        self.LOW_DIR =  self.BASE_DIR + 'low'
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
for train, test in kfold.split(x, y):
    sm = SMOTE(random_state=42, kind='svm')
    # adasyn = ADASYN(random_state=42)
    x_res, y_res = sm.fit_resample(x[train], y[train])
    gs.fit(x_res, y_res)
    # gs.fit(x[train], y[train])
    print('Best score', gs.best_score_)
    print('Best model', gs.best_estimator_)

    pred = gs.predict(x[test])
    print(confusion_matrix(y[test], pred))
    precision.append(precision_score(y[test], pred, average="weighted"))
    recall.append(recall_score(y[test], pred, average="weighted"))
    f1_sc.append(f1_score(y[test], pred, average="weighted"))

print("precision:{:.2f}".format(np.mean(precision)))
print("recall:{:.2f}".format(np.mean(recall)))
print("f1_score:{:.2f}".format(np.mean(f1_sc)))