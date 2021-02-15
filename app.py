import os
import csv
import numpy as np
from sklearn import decomposition
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import extractFeatures
from keras.models import load_model
import joblib
import pickle

project_root = os.path.dirname(os.path.realpath('__file__'))
template_path = os.path.join(project_root, 'templates')
static_path = os.path.join(project_root, 'static')
app = Flask(__name__, template_folder=template_path, static_folder=static_path)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


class PredForm(Form):
    sequence = TextAreaField(u'Enter Sequence:', [validators.DataRequired()])


def SimpleParser(sequence):
    seq = sequence.split('\n')
    re = ''
    for x in seq:
        re = re + x[:len(x) - 2]
    return re


@app.route("/", methods=['GET', 'POST'])
def index():
    form = PredForm(request.form)
    print(form.errors)
    if request.method == 'POST':
        input_seq = request.form['sequence']
        if input_seq.isalpha():
            featur = extractFeatures.get_features([input_seq])
            np.random.seed(5)
            inputSize = 194
            outputCol = inputSize + 1

            dataset = np.genfromtxt("./FVs.csv", delimiter=",", dtype=float)
            # print(dataset.shape)
            X = np.array(dataset[:, 0:inputSize], dtype=np.float32)
            X = np.nan_to_num(X)
            print(featur)
            Y = dataset[:, inputSize:outputCol]
            # print(min(Y))
            # print(max(Y))
            # print(Y.ravel())
            Y = np.nan_to_num(Y)
            std_scale = StandardScaler().fit(X)
            X = std_scale.transform(X)
            X = np.array(X, dtype=np.float32)
            X = np.nan_to_num(X)

            pca = decomposition.PCA(n_components=2)
            pca.fit(X)
            X = pca.transform(X)
            clf = RandomForestClassifier(n_estimators=50, oob_score=True, n_jobs=-1, warm_start=True).fit(X, Y.ravel())
            # joblib.dump(clf, 'model.pkl')
            model = joblib.load('model.pkl')

            pred = np.round(clf.predict(X))

            featur = np.array(featur, dtype=np.float32)
            featur = np.nan_to_num(featur)
            featur = std_scale.transform(featur)
            featur = np.nan_to_num(featur)
            featur = pca.transform(featur)
            # r = clf.predict_proba(featur)
            # r = clf.predict(featur)
            r = model.predict(featur)

            # for i in r:
            #     seq=i
            # print(seq)
            print(r[0])
            if r[0] == 1.0:
                class1 = 'POSITIVE SUMOYLATION SEQUENCE'
            if r[0] == 0.0:
                class1 = 'NEGATIVE SUMOYLATION SEQUENCE'

            result = [input_seq, class1]

            return resultPage(result)
        else:
            result=['Invalid Sequence','aaa']
            return resultPage(result)

    return render_template('home.html', form=form, title="Home")


def resultPage(result):
    return render_template('result.html', result=result, title="Results")


if __name__ == "__main__":
    app.run()
