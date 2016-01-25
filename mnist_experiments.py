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

BASENAME = "ckn-mnist-full"

OUTPATH = "/work/vaishaal/ckm/{0}".format(BASENAME)
td = lambda x : os.path.join(BASENAME, x)


ONE_LEVEL_FEATURE_CONFIGS = {"1level.1" : { 'dataset' : ['mnist_full'], 'seed': [0], 'gammas': [[0.8],[1.2], [1.6], [2.0], [2.4]], 'filters': [[50]], 'patch_shapes': [[5]]}}

SOLVE_CONFIGS = {'loss': ['log','hinge'], 'reg': [0.01,0.001,0.0001]}

def dict_product(dicts):
    return (dict(itertools.izip(dicts, x)) for x in itertools.product(*dicts.itervalues()))

def feature_params():
    default = {}
    for exp_name, exp_config in ONE_LEVEL_FEATURE_CONFIGS.iteritems():
        for ei, d in enumerate(dict_product(exp_config)):
            dc = default.copy()
            dc.update(d)
            infile = None
            fname = "features.%s.%04d" % (exp_name, ei)
            outfile = td(fname + ".pickle")
            featureLocation = os.path.join(OUTPATH, fname)
            yield infile, outfile, dc

def solve_params():
    default = {}
    for exp_name, exp_config in ONE_LEVEL_FEATURE_CONFIGS.iteritems():
        for ei, d in enumerate(dict_product(exp_config)):
            infile = td("features.%s.%04d.pickle" % (exp_name, ei)) # add pickle
            for si, s in enumerate(dict_product(SOLVE_CONFIGS)):
                outfile = "solve.%s.%d_%s.%d.pickle" % (exp_name, si, exp_name, ei)
                yield infile, td(outfile), outfile, exp_name, s, si



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
                 'X_train' : X_train_lift.reshape(X_train_lift.shape[0], -1),
                 'X_test'  : X_test_lift.reshape(X_test_lift.shape[0], -1),
                 'y_train' : y_train,
                 'y_test'  : y_test,
                 'config' : dc }, open(outfile, 'w'))



@follows(run_features)
@files(solve_params)
def run_solve(infile, outfile, outloc, exp_name, solve_config, si):
    feature_out = pickle.load(open(infile, 'r'))
    solve_out = feature_out.copy()
    f_config = feature_out['config']
    loss = solve_config['loss']
    reg = solve_config['reg']
    rand = f_config['seed']
    clf = SGDClassifier(loss=loss, alpha=reg, random_state=rand)
    print feature_out['X_train'].shape
    clf.fit(feature_out['X_train'], feature_out['y_train'])
    X_test = feature_out['X_test']
    y_pred = clf.predict(X_test)
    solve_out['y_pred'] = y_pred
    y_test = solve_out['y_test'] = feature_out['y_test']
    acc = metrics.accuracy_score(y_test, y_pred)
    print("Reg is {0}, gamma is {1}, loss is {2} accuracy is: {3}".format(reg, f_config['gammas'], loss, acc))
    pickle.dump({'infile' : infile,
                 'outfile' : outfile,
                 'solve_out' : solve_out,
                 'fc_dict' : f_config,
                 'solve_config' : solve_config,
                 'si' : si},
                open(outfile, 'w'), -1)



if __name__ == "__main__":
    pipeline_run([run_features, run_solve])


