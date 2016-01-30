import argparse

from yaml import load, dump

from pandas import DataFrame
from ckm import *

import logging
import collections

from tabulate import tabulate

from softmax import softmax

from sklearn import metrics

'''
Driver for Convolutional Kernel Machine experiments
'''

def main():
    # First parse arguments for the CKM
    parser = argparse.ArgumentParser(description='Convolutional Kernel Machine')
    parser.add_argument('config', help='path to config file for experiment')
    args = parser.parse_args()
    exp = parse_experiment(args.config)
    logging.info('Experiment mode: {0}'.format(exp.get("mode")))
    if (exp.get("mode") == "python"):
        results = python_run(exp)
    print tabulate(results, headers="keys")

def load_data(dataset="mnist_small"):
    '''
        @param dataset: The dataset to load
        @param random_state: random state to control random parameter

        Load a specified dataset currently only
        "mnist_small" and "mnist" are supported
    '''
    if (dataset == "mnist_small"):
        X_train = np.loadtxt("./mldata/mnist_small/X_train", delimiter=",").reshape(1540,64)
        X_test = np.loadtxt("./mldata/mnist_small/X_test", delimiter=",").reshape(257,64)
        y_train = np.loadtxt("./mldata/mnist_small/y_train", delimiter=",")
        y_test = np.loadtxt("./mldata/mnist_small/y_test", delimiter=",")
    elif dataset == "mnist_full":
        mndata = MNIST('./mldata/mnist')
        X_train, y_train = map(np.array, mndata.load_training())
        X_test, y_test = map(np.array, mndata.load_testing())
        X_train = X_train/255.0
        X_test = X_test/255.0
    else:
        raise Exception("Datset not found")

    X_train = X_train[:,:,np.newaxis]
    X_test = X_test[:,:,np.newaxis]
    return (X_train, y_train), (X_test, y_test)


def flatten_dict(d, parent_key='', sep='_'):

    ''' Borrowed from:
    http://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
    '''
    items = []
    for k, v in d.items():
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, k, sep=sep).items())
        else:
            items.append((k, v))
    return dict(items)


def python_run(exp):
    if (exp.get("seed") == None):
        exp["seed"] = int(random.random*(2*32))
    dataset = exp["dataset"]
    seed = exp["seed"]
    (X_train, y_train), (X_test, y_test) = load_data(dataset)
    ckm_params = exp["ckm_params"]
    solver_params = exp["solver_params"]
    X_train_lift, X_test_lift = gen_features(ckm_params, X_train, X_test, seed)
    y_train_pred, y_test_pred = solve(solver_params, X_train_lift, y_train, X_test_lift, y_test, seed)
    results = compute_metrics(exp, y_train, y_train_pred, y_test, y_test_pred)
    return results

def scala_run(exp):
    raise(Exception("Scala Run has not been implemented"))

def gen_features(ckm_params, X_train, X_test, seed):
    ckm_run = build_ckm(ckm_params, seed)
    X_train_lift, X_test_lift = ckm_run(X_train, X_test)
    return X_train_lift, X_test_lift

def compute_metrics(exp, y_train, y_train_pred, y_test, y_test_pred):
    exp_flatten = flatten_dict(exp)
    exp_flatten = dict(map(lambda x: (x[0], [str(x[1])]), exp_flatten.items()))
    df = DataFrame.from_dict(exp_flatten)
    train_acc = metrics.accuracy_score(y_train, y_train_pred)
    test_acc = metrics.accuracy_score(y_test, y_test_pred)
    df.insert(len(df.columns), "train_acc", train_acc)
    df.insert(len(df.columns), "test_acc", test_acc)
    return df


def build_ckm(ckm_params, seed):
    layers = ckm_params.get("layers")
    filters = ckm_params.get("filters")
    bandwidth = ckm_params.get("bandwidth")
    patch_sizes = ckm_params.get("patch_sizes")
    def ckm_run(X_train, X_test):
        for i in range(layers):
            patch_shape = (patch_sizes[i], patch_sizes[i])
            X_train, X_test =  ckm_apply(X_train, X_test, patch_shape, bandwidth[i], filters[i], random_state=(seed+i))
        return X_train, X_test
    return ckm_run

def solve(solve_params, X_train, y_train, X_test, y_test, seed):
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    loss = solve_params["loss"]
    reg = solve_params["reg"]
    if (loss == "softmax"):
        y_train_pred, y_test_pred = softmax(X_train, y_train, X_test, y_test, reg)
    else:
        clf = SGDClassifier(loss=loss, random_state=RANDOM_STATE, alpha=reg)
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
    return y_train_pred, y_test_pred

def parse_experiment(config_file):
    return load(open(config_file))

if __name__ == "__main__":
    main()
