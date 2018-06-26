import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def select_model(dataset, target, variables, models, scoring, seed):
    msgs = []
    results = []
    names = []
    x, y = dataset.loc[:, variables], dataset.loc[:, target]
    for name, model in models:
        kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        cv_results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = [name, cv_results.mean(), cv_results.std()]
        msgs.append(msg)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    msgs = pd.DataFrame(msgs, columns=['name', 'mean', 'std'])
    return msgs


def train_model(training, test, target, variables, model):
    x, y = training.loc[:, variables], training.loc[:, target]
    model.fit(x, y)
    predictions = model.predict(test.loc[:, variables])
    pickle.dump(model, open(os.path.join(os.getcwd(), "models", "titanic"), 'wb'))
    return predictions
