from sklearn import datasets
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation
from sklearn import metrics
import datetime
import time
from ruffus import *

st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
BASENAME = "vaishaal-ckn-{0}".format(st)
DELIM = "="*100

def _load_mnist(small=True):
    if small:
        mnist = datasets.load_digits()
    else:
        mnist = fetch_mldata('MNIST original', data_home="/Users/vaishaal/research/ckm")

    return mnist.data, mnist.target

def load_data(dataset="mnist_small"):
    return _load_mnist()



if __name__ == "__main__":
    X, y = load_data()
    print(DELIM)
    print("Data load complete")
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5, random_state=0)
    clf = SGDClassifier(loss="hinge", random_state=0)
    clf.fit(X_train, y_train)
    print(DELIM)
    print("Classifier Training complete")
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(DELIM)
    print("Test accuracy: {0}".format(acc))



