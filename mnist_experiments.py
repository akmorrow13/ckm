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

import itertools
import os

from ckm import *

import pickle

st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
BASENAME = "vaishaal-ckn-{0}".format(st)
DELIM = "="*100
RANDOM_STATE=0
flatten = ndarray.flatten

BASENAME = "ckn-mnist"

OUTPATH = "/work/vaishaal/ckm/{0}".format(BASENAME)
td = lambda x : os.path.join(BASENAME, x)

FEATURE_CONFIGS = {"2level.1" : { 'dataset' : ['mnist_small'], 'seed': [0], 'gammas': [[1.6, 1.6]], 'filters': [[50, 200]], 'patch_shapes': [[5, 3]]}}

def dict_product(dicts):
    return (dict(itertools.izip(dicts, x)) for x in itertools.product(*dicts.itervalues()))

def feature_params():
    default = {}
    for exp_name, exp_config in FEATURE_CONFIGS.iteritems():
        for ei, d in enumerate(dict_product(exp_config)):
            dc = default.copy()
            dc.update(d)
            infile = None
            fname = "features.%s.%04d" % (exp_name, ei)
            outfile = td(fname + ".pickle")
            featureLocation = os.path.join(OUTPATH, fname)
            yield infile, outfile, dc




@mkdir(BASENAME)
@files(feature_params)
def run_features(infile, outfile, dc):
    (X_train, y_train), (X_test, y_test) = load_data(dc["dataset"])
    X_train_lift = X_train[:,:,np.newaxis]
    X_test_lift = X_test[:,:,np.newaxis]
    t1 = time.time()
    for i, gamma in enumerate(dc["gammas"]):
        filter_size = dc["filters"][i]
        ps = dc["patch_shapes"][i]
        patch_shape = (ps, ps)
        X_train_lift, X_test_lift = ckm_apply(X_train_lift, X_test_lift, patch_shape,gamma , filter_size, True)
        X_train_lift = X_train_lift.reshape(X_train_lift.shape[0], -1, filter_size)
        X_test_lift = X_test_lift.reshape(X_test_lift.shape[0], -1, filter_size)

    t2 = time.time()

    pickle.dump({"time" : t2-t1,
                 'infile' : infile,
                 'outfile' : outfile,
                 'dataset' : dc['dataset'],
                 'X_train' : X_train_lift,
                 'X_test'  : X_test_lift,
                 'y_train' : y_train,
                 'y_test'  : y_test,
                 'config' : dc }, open(outfile, 'w'))

if __name__ == "__main__":
    pipeline_run([run_features])


