import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from parseValidFiles import *
from gensim import models
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE


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
print(x.shape, y.shape)
sm = SMOTE(random_state=42)
x_res, y_res = sm.fit_resample(x, y)
print(x_res.shape, y_res.shape)
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
parameters = {'gamma': [0.01, 0.02, 0.05, 0.1, 0.2, 1.0], 'degree': [1, 2, 3]}
gs = GridSearchCV(svm.SVC(kernel='poly', class_weight="balanced", probability=True), parameters, cv=5)

for train, test in kfold.split(x_res, y_res):
    gs.fit(x_res[train], y_res[train])
    print('Best score', gs.best_score_)

    pred = gs.predict(x_res[test])
    print(classification_report(y_res[test], pred))

    prob = gs.predict_proba(x_res[test])
    fpr, tpr, thresholds = roc_curve(y_res[test], prob)
    precision, recall, thresholds = precision_recall_curve(y_res[test], prob)
    area = auc(recall, precision)
    print("auc:", area)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.title("ROC curve")
    plt.show()

