from sklearn import datasets
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn import cross_validation
from sklearn import metrics
from skimage.measure import block_reduce
from scipy import signal

import scipy
import numpy as np
from numpy import ndarray

import datetime
import time
from ruffus import *
from mnist import MNIST


st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
BASENAME = "vaishaal-ckn-{0}".format(st)
DELIM = "="*100
RANDOM_STATE=0
flatten = ndarray.flatten

def load_data(dataset="mnist_small", random_state=RANDOM_STATE):
    '''
        @param dataset: The dataset to load
        @param random_state: random state to control random parameter

        Load a specified dataset currently only
        "mnist_small" and "mnist" are supported
        if the data set does not come split up as train and test
        a random subset is chosen as the "test" set
    '''
    if (dataset == "mnist_small"):
        mnist = datasets.load_digits()
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(mnist.data, mnist.target, test_size=1.0/7.0, random_state=random_state)
    elif dataset == "mnist_full":
        mndata = MNIST('./mldata')
        X_train, y_train = map(np.array, mndata.load_training())
        X_test, y_test = map(np.array, mndata.load_testing())
        X_train = X_train/255.0
        X_test = X_test/255.0
    elif dataset == "mnist_full_sklearn":
        mnist = fetch_mldata('MNIST original', data_home="/work/vaishaal/ckm")
        mnist.data = mnist.data/255.0
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(mnist.data, mnist.target, test_size=1.0/7.0, random_state=random_state)
    else:
        raise Exception("Datset not found")
    return (X_train, y_train), (X_test, y_test)

def msg(message, delim=False):
    if (delim):
        print(DELIM)
    print(message)

def apply_patch_rbf(X,imsize, patches, rbf_weights, rbf_offset):

    new_shape = patches.shape[:-3] + (patches.shape[-3]*patches.shape[-1]*patches.shape[-2],)
    patches = patches.reshape(new_shape)
    print "new patch shape", patches.shape
    X_lift = np.zeros((X.shape[0], X.shape[1], len(rbf_offset)))
    k = 0
    for n in range(X.shape[0]):
        image_patches = patches[n, :]
        flat_patch_norm = np.maximum(np.linalg.norm(image_patches, axis=1), 0.001)[:,np.newaxis]
        flat_patch_normalized = image_patches/flat_patch_norm
        projection = flat_patch_normalized.dot(rbf_weights)
        projection += rbf_offset
        np.cos(projection, projection)
        projection *= np.sqrt(2.) / np.sqrt(len(rbf_offset))
        X_lift[n] = flat_patch_norm*projection

    X_lift = X_lift.reshape(X_lift.shape[0], X_lift.shape[1]*X_lift.shape[2])
    # Contrast normalization
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


def get_model_acc(clf, X_train, y_train, X_test, y_test, r_state=RANDOM_STATE):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

def patchify_all_imgs(X, patch_shape, pad=True, pad_mode='constant', cval=0):
    out = []
    for x in X:
        dim = x.shape[0]
        x = x.reshape(int(np.sqrt(dim)), int(np.sqrt(dim)), x.shape[1])
        patches = patchify(x, patch_shape)
        out.append(patches.reshape(dim, patch_shape[0], patch_shape[1], -1))
    return np.array(out)

def patchify(img, patch_shape, pad=True, pad_mode='constant', cval=0):
    ''' Function borrowed from:
    http://stackoverflow.com/questions/16774148/fast-way-to-slice-image-into-overlapping-patches-and-merge-patches-to-image
    '''
    #FIXME: Make first two coordinates of output dimension shape as img.shape always

    if pad:
        pad_size= (patch_shape[0]/2, patch_shape[0]/2)
        img = np.pad(img, (pad_size, pad_size, (0,0)),  mode=pad_mode, constant_values=cval)

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
def ckm_apply(X_train, X_test, patch_shape, gamma, n_components, pool=True):
    patch_rbf = RBFSampler(gamma=gamma, random_state=RANDOM_STATE, n_components=n_components).fit(np.zeros((1,patch_shape[0]*patch_shape[1]*X_train.shape[-1])))
    patches_train = patchify_all_imgs(X_train, patch_shape)
    print "Generated train patches"
    print "Patches_train size", patches_train.shape
    print "RBF map shape", patch_rbf.random_weights_.shape
    X_patch_lift_train = apply_patch_rbf(X_train, X_train.shape[1], patches_train, patch_rbf.random_weights_, patch_rbf.random_offset_)
    print "Lifted train"
    patches_test = patchify_all_imgs(X_test, patch_shape)
    print "Generated test patches"
    X_patch_lift_test = apply_patch_rbf(X_test, X_test.shape[1], patches_test, patch_rbf.random_weights_, patch_rbf.random_offset_)
    print "Lifted test"
    print "Pre pool shape", X_patch_lift_train.shape

    imsize = (np.sqrt(X_train.shape[1]), np.sqrt(X_train.shape[1]), patch_rbf.n_components)
    if (pool):
        X_pooled_train = np.apply_along_axis(lambda im: gaussian_pool(im,2, imsize), 1, X_patch_lift_train)
        X_pooled_test = np.apply_along_axis(lambda im: gaussian_pool(im,2, imsize), 1, X_patch_lift_test)
        return X_pooled_train, X_pooled_test
    else:
        return X_patch_lift_train, X_patch_lift_test

if __name__ == "__main__":
    msg("Start Data Load", True)
    (X_train, y_train), (X_test, y_test) = load_data("mnist_full")
    msg("Data load complete")

    msg("Start linear model train", True)
    clf = SGDClassifier(loss="hinge", random_state=RANDOM_STATE)
    np.lib.stride_tricks
    msg("Linear Classifier Validation accuracy: {0}".format(get_model_acc(clf, X_train, y_train, X_test, y_test)))

    msg("Start Random RBF model train", True)
    rbf_feature = RBFSampler(gamma=0.01, random_state=RANDOM_STATE, n_components=500).fit(X_train)
    X_train_lift = rbf_feature.transform(X_train)
    X_test_lift = rbf_feature.transform(X_test)
    msg("Random RBF Classifier Validation accuracy: {0}".format(get_model_acc(clf, X_train_lift, y_train, X_test_lift, y_test)))

    clf = SGDClassifier(loss="hinge", alpha=0.002, random_state=RANDOM_STATE)
    msg("Start Random Patch RBF train", True)
    patch_shape = (5,5)
    patch_shape_2 = (3,3)

    X_train = X_train[:,:,np.newaxis]
    X_test = X_test[:,:,np.newaxis]

    X_train_l1, X_test_l1 = ckm_apply(X_train, X_test, patch_shape,1.616 , 50, True)

    X_train_l1 = X_train_l1.reshape(X_train.shape[0],-1, 50)
    X_test_l1 = X_test_l1.reshape(X_test.shape[0],-1, 50)

    X_train_l2, X_test_l2 = ckm_apply(X_train_l1, X_test_l1, patch_shape_2, 1.616, 200, True)

    msg("Level 2 Patch RBF Classifier Test accuracy: {0}".format(get_model_acc(clf, X_train_l2, y_train, X_test_l2, y_test)))


