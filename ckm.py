from sklearn import datasets
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn import cross_validation
from sklearn import metrics

import numpy as np
from numpy import ndarray

import datetime
import time
from ruffus import *

st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
BASENAME = "vaishaal-ckn-{0}".format(st)
DELIM = "="*100
RANDOM_STATE=0
flatten = ndarray.flatten

def _load_mnist(small=True):
    if small:
        mnist = datasets.load_digits()
    else:
        mnist = fetch_mldata('MNIST original', data_home="/Users/vaishaal/research/ckm")

    return mnist.data, mnist.target

def load_data(dataset="mnist_small"):
    return _load_mnist()

def apply_patch_rbf(X, patch_shape, rbf):
    print(DELIM)
    print("Applying Patch RBF")
    print("X Input Shape: {0}".format(X.shape))
    patches  = patchify(X, patch_shape)

    print("Patch Shape: {0}".format(patches.shape))
    print patches[0,0].shape
    X_lift = np.zeros((X.shape[0], X.shape[1], rbf.n_components))
    for n in range(X.shape[0]):
        for i in range(64):
            X_lift[n,i] = rbf.transform(flatten(patches[n, i]))
    X_lift = X_lift.reshape(X_lift.shape[0], X_lift.shape[1]*X_lift.shape[2])
    print("X Output Shape: {0}".format(X_lift.shape))
    return X_lift


def patchify(img, patch_shape, pad=True, pad_mode='constant', cval=0):
    ''' Function borrowed from:
    http://stackoverflow.com/questions/16774148/fast-way-to-slice-image-into-overlapping-patches-and-merge-patches-to-image
    '''
    if pad:
        img = np.pad(img, patch_shape[0]/2, mode=pad_mode, constant_values=cval)

    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y = img.shape
    x, y = patch_shape
    shape = ((X-x+1), (Y-y+1), x, y) # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
#    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([Y, 1, Y, 1])
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    return patches
if __name__ == "__main__":
    X, y = load_data()
    print(DELIM)
    print("Data load complete")
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5, random_state=0)
    clf = SGDClassifier(loss="hinge", random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    print(DELIM)
    print("Linear Classifier Training complete")
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(DELIM)
    print("Linear Classifier Test accuracy: {0}".format(acc))
    print(DELIM)
    rbf_feature = RBFSampler(gamma=0.001, random_state=RANDOM_STATE, n_components=5000).fit(X)
    clf2 = SGDClassifier(loss="hinge", random_state=RANDOM_STATE)
    clf2.fit(rbf_feature.transform(X_train), y_train)
    print("Random RBF Classifier Training complete")
    y_pred2 = clf2.predict(rbf_feature.transform(X_test))
    acc2 = metrics.accuracy_score(y_test, y_pred2)
    print(DELIM)
    print("Random RBF Classifier Test accuracy: {0}".format(acc2))
    print(DELIM)
    print("APPLY PATCH RBF")
    patch_shape = (5,5)
    patch_rbf = RBFSampler(gamma=0.001, random_state=RANDOM_STATE, n_components=100).fit(np.zeros(patch_shape[0]*patch_shape[1]))
    X_train_lift = apply_patch_rbf(X_train, patch_shape, patch_rbf)
    print(DELIM)
    print("PATCH RBF COMPLETE")

    clf3 = SGDClassifier(loss="hinge", random_state=RANDOM_STATE)
    clf3.fit(X_train_lift, y_train)
    print("Patch RBF Classifier Training complete")
    y_pred3 = clf3.predict(apply_patch_rbf(X_test, patch_shape, patch_rbf))
    acc3 = metrics.accuracy_score(y_test, y_pred3)
    print(DELIM)
    print("Patch RBF Classifier Test accuracy: {0}".format(acc3))





