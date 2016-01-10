from sklearn import datasets
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn import cross_validation
from sklearn import metrics
from skimage.measure import block_reduce

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

def msg(message, delim=False):
    if (delim):
        print(DELIM)
    print(message)

def apply_patch_rbf(X, patch_shape, rbf):

    msg("Applying Patch RBF", True)
    msg("X Input Shape: {0}".format(X.shape))
    patches  = patchify(X, patch_shape)

    msg("Patch Shape: {0}".format(patches.shape))
    X_lift = np.zeros((X.shape[0], X.shape[1], rbf.n_components))
    for n in range(X.shape[0]):
        for i in range(64):
            X_lift[n,i] = rbf.transform(flatten(patches[n, i]))
    X_lift = X_lift.reshape(X_lift.shape[0], X_lift.shape[1]*X_lift.shape[2])
    msg("X Output Shape: {0}".format(X_lift.shape))
    return X_lift

def pool(x, pool_size, imsize, func=np.sum):
    x = x.reshape(imsize)
    x_pool = block_reduce(x, block_size=  pool_size + (1,), func=func)
    return flatten(x_pool)





def get_model_acc(clf, X, y, r_state=RANDOM_STATE):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5, random_state=r_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

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
    msg("Start Data Load", True)
    X, y = load_data()
    msg("Data load complete")

    msg("Start linear model train", True)
    clf = SGDClassifier(loss="hinge", random_state=RANDOM_STATE)
    msg("Linear Classifier Validation accuracy: {0}".format(get_model_acc(clf, X, y)))


    msg("Start Random RBF model train", True)
    rbf_feature = RBFSampler(gamma=0.001, random_state=RANDOM_STATE, n_components=5000).fit(X)
    X_lift = rbf_feature.transform(X)
    msg("Random RBF Classifier Validation accuracy: {0}".format(get_model_acc(clf, X_lift, y)))


    msg("Start Random Patch RBF train", True)
    patch_shape = (5,5)
    patch_rbf = RBFSampler(gamma=0.001, random_state=RANDOM_STATE, n_components=5000).fit(np.zeros(patch_shape[0]*patch_shape[1]))
    X_patch_lift = apply_patch_rbf(X, patch_shape, patch_rbf)

    msg("Patch RBF Classifier Test accuracy: {0}".format(get_model_acc(clf, X_patch_lift, y)))
    imsize = (np.sqrt(X.shape[1]), np.sqrt(X.shape[1]), patch_rbf.n_components)
    X_pooled = np.apply_along_axis(lambda im: pool(im,(2,2), imsize), 1, X_patch_lift)

    msg("pooled Patch RBF Classifier Test accuracy: {0}".format(get_model_acc(clf, X_pooled, y)))



