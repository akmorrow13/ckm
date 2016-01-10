from sklearn import datasets
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation
from sklearn import metrics
from sklearn.kernel_approximation import Nystroem
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
    print("Linear Classifier Training complete")
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(DELIM)
    print("Linear Classifier Test accuracy: {0}".format(acc))

    print(DELIM)
    clf2 = SGDClassifier(loss="hinge", random_state=0)
    nystrom_feature = Nystroem(gamma=0.0003, random_state=1, n_components=500).fit(X_train)
    clf2.fit(nystrom_feature.transform(X_train), y_train)
    print("Nystroem Classifier Training complete")
    y_pred2 = clf2.predict(nystrom_feature.transform(X_test))
    acc2 = metrics.accuracy_score(y_test, y_pred2)
    print(DELIM)
    print("Nystroem Classifier Test accuracy: {0}".format(acc2))


