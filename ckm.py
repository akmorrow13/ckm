from sklearn import datasets
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn import cross_validation
from sklearn import metrics
from skimage.measure import block_reduce
from scipy import signal

import scipy

from numba import jit

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
        mnist.data = mnist.data/255.0

    return mnist.data, mnist.target

def load_data(dataset="mnist_small"):
    if (dataset == "mnist_small"):
        data = _load_mnist(True)
    elif dataset == "mnist_full":
        data = _load_mnist(False)
    return data

def msg(message, delim=False):
    if (delim):
        print(DELIM)
    print(message)

def apply_patch_rbf(X,imsize, patch_shape, rbf):

    msg("Applying Patch RBF", True)
    msg("X Input Shape: {0}".format(X.shape))
    patches  = patchify(X, patch_shape)
    new_shape = patches.shape[:-3] + (patches.shape[-3]*patches.shape[-1]*patches.shape[-2],)
    print "OLD SHAPE", patches.shape
    print "NEW SHAPE", new_shape
    patches = patches.reshape(new_shape)
    msg("Patch Shape: {0}".format(patches.shape))
    X_lift = np.zeros((X.shape[0], X.shape[1], rbf.n_components))
    k = 0
    w = rbf.random_weights_[np.newaxis, np.newaxis,:,:]
    print "W shape", w.shape

    for n in range(X.shape[0]):
        image_patches = patches[n, :]
        flat_patch_norm = np.maximum(np.linalg.norm(image_patches, axis=1), 0.001)[:,np.newaxis]
        flat_patch_normalized = image_patches/flat_patch_norm
        X_lift[n] = flat_patch_norm*rbf.transform(flat_patch_normalized)

    X_lift = X_lift.reshape(X_lift.shape[0], X_lift.shape[1]*X_lift.shape[2])
    # Contrast normalization
    msg("X Output Shape: {0}".format(X_lift.shape))
    return X_lift

def pool(x, pool_size, imsize, func=np.average):
    x = x.reshape(imsize)
    spatial_indices = np.indices(x.shape[:2])
    x_pool = block_reduce(x, block_size = pool_size + (1,), func=func)
    return flatten(x_pool)


def gaussian_pool(x, pool_size, imsize):
    x = x.reshape(imsize)
    gauss = signal.gaussian(6, 1/np.sqrt(np.sqrt(2)))
    kernel = np.outer(gauss, gauss)
    x_out =  x.copy()
    for i in range(x_out.shape[2]):
        x_out[:,:,i] = signal.fftconvolve(x[:,:,i], kernel, mode='same')
    return flatten(x_out[::pool_size, ::pool_size,:])


def get_model_acc(clf, X, y, r_state=RANDOM_STATE):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=1.0/7.0, random_state=r_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

def patchify(img, patch_shape, pad=True, pad_mode='constant', cval=0):
    ''' Function borrowed from:
    http://stackoverflow.com/questions/16774148/fast-way-to-slice-image-into-overlapping-patches-and-merge-patches-to-image
    '''
    #FIXME: Make first two coordinates of output dimension shape as img.shape always

    if pad:
        pad_size= (patch_shape[0]/2, patch_shape[0]/2)
        img = np.pad(img, [pad_size, pad_size, (0,0)],  mode=pad_mode, constant_values=cval)

    img = np.ascontiguousarray(img)  # won't make a copy if not needed

    X, Y, Z = img.shape
    x, y= patch_shape
    shape = ((X-x+1), (Y-y+1), x, y, Z) # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
#    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([Y*Z, Z, Y*Z, Z, 1])
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    return patches
if __name__ == "__main__":
    msg("Start Data Load", True)
    X, y = load_data("mnist_small")
    msg("Data load complete")

    msg("Start linear model train", True)
    clf = SGDClassifier(loss="hinge", random_state=RANDOM_STATE)
    np.lib.stride_tricks
    msg("Linear Classifier Validation accuracy: {0}".format(get_model_acc(clf, X, y)))


    msg("Start Random RBF model train", True)
    rbf_feature = RBFSampler(gamma=0.0095, random_state=RANDOM_STATE, n_components=500).fit(X)
    X_lift = rbf_feature.transform(X)
    msg("Random RBF Classifier Validation accuracy: {0}".format(get_model_acc(clf, X_lift, y)))


    msg("Start Random Patch RBF train", True)
    patch_shape = (5,5)
    patch_rbf = RBFSampler(gamma=1.7, random_state=RANDOM_STATE, n_components=50).fit(np.zeros(patch_shape[0]*patch_shape[1]))
    X_patch_lift = apply_patch_rbf(X[:,:,np.newaxis], X.shape[1], patch_shape, patch_rbf)
    msg("Patch RBF Classifier Test accuracy: {0}".format(get_model_acc(clf, X_patch_lift, y)))

    '''
    imsize = (np.sqrt(X.shape[1]), np.sqrt(X.shape[1]), patch_rbf.n_components)
    X_pooled = np.apply_along_axis(lambda im: gaussian_pool(im,2, imsize), 1, X_patch_lift)
    msg("pooled Patch RBF Classifier Test accuracy: {0}".format(get_model_acc(clf, X_pooled, y)))

    patch_shape2 = (3,3)
    patch_rbf2 = RBFSampler(gamma=1, random_state=RANDOM_STATE, n_components=2000).fit(np.zeros(patch_shape2[0]*patch_shape2[1]*50))
    X_patch_layer2 = apply_patch_rbf(X_pooled.reshape(1797, 196, 50), 16, patch_shape2, patch_rbf2)
    msg("2 level CKN Classifier Test accuracy: {0}".format(get_model_acc(clf, X_patch_layer2, y)))
    '''
