import argparse

from yaml import load, dump

import features_pb2
from pandas import DataFrame
from ckm import *
import glob 

import logging
import collections
import os
import subprocess

from tabulate import tabulate

from softmax import softmax

from sklearn import metrics
from sklearn.linear_model import SGDClassifier

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
    print exp
    if (exp.get("mode") == "python"):
        results = python_run(exp)
    elif (exp.get("mode") == "scala"):
        results = scala_run(exp, args.config)
    if (not (results is None)):
        print tabulate(results, headers="keys")

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
    verbose = exp["verbose"]
    center = exp.get("center")
    (X_train, y_train), (X_test, y_test) = load_data(dataset, center)
    if (verbose):
        print "Data loaded Train shape: {0}, Test Shape: {1}, Train Labels shape: {2}, \
        Test Labels shape {3}".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    X_train_lift, X_test_lift = gen_features(exp, X_train, X_test, seed)
    if (verbose):
        print "Data Featurized Train shape: {0}, Test Shape: {1}, Train Labels shape: {2}, \
        Test Labels shape {3}".format(X_train_lift.shape, X_test_lift.shape, y_train.shape, y_test.shape)

    y_train_pred, y_test_pred = solve(exp, X_train_lift, y_train, X_test_lift, y_test, seed)
    results = compute_metrics(exp, y_train, y_train_pred, y_test, y_test_pred)
    return results

def scala_run(exp, yaml_path):
    expid = exp["expid"]
    config_yaml = yaml_path
    env = os.environ.copy()
    # sanity check before running the process

    # if not os.path.isdir(outdir):
    #     raise ValueError("output dir must exist")

    logfile = expid + ".spark.log"
    # if os.path.exists(logfile) and output_sanity_check:
    #     raise ValueError("output dir has logfile, should be empty")
    pipelineClass="pipelines.CKM"
    pipelineJar = "/home/eecs/vaishaal/ckm/keystone_pipeline/target/scala-2.10/ckm-assembly-0.1.jar"
    if not os.path.exists(pipelineJar):
        raise ValueError("Cannot find pipeline jar")

    # basically capturing the popen output and saving it to disk  and
    # printing it to screen are a multithreaded nightmare, so we let
    # tee do the work for us
    yarn = exp.get("yarn")
    if (yarn):
        p = subprocess.Popen(" ".join(["./keystone_pipeline/bin/run-pipeline-yarn.sh", pipelineClass,
                                   pipelineJar, config_yaml]),
                         shell=True, executable='/bin/bash')
    else:
        p = subprocess.Popen(" ".join(["./keystone_pipeline/bin/run-pipeline.sh", pipelineClass,
                                   pipelineJar, config_yaml]),
                         shell=True, executable='/bin/bash')
    #p = subprocess.Popen(cmd, shell=True, executable='/bin/bash', env = env)
    p.wait()

    if p.returncode != 0:
        raise Exception("invocation terminated with non-zero exit status")
    if (exp["solve"]):
        y_train, y_train_weights  = load_scala_results("/tmp/ckm_train_results")
        y_test, y_test_weights  = load_scala_results("/tmp/ckm_test_results")
        # TODO: Do more interesting things here
        y_train_pred = np.argmax(y_train_weights, axis=1)
        y_test_pred = np.argmax(y_test_weights, axis=1)
        results = compute_metrics(exp, y_train, y_train_pred, y_test, y_test_pred)
        return results
    else:
        return None


def gen_features(exp, X_train, X_test, seed):
    ckm_run = build_ckm(exp, seed)
    X_train_lift, X_test_lift = ckm_run(X_train, X_test)
    X_train = X_train_lift.reshape(X_train.shape[0], -1)
    X_test = X_test_lift.reshape(X_test.shape[0], -1)
    return X_train, X_test

def save_features_python(X, y, name):
    X = X.reshape(X.shape[0], -1)
    dataset = features_pb2.Dataset()
    dataset.name = name
    for i in range(len(y)):
        datum = dataset.data.add()
        datum.label = int(y[i])
        datum.data.extend(X[i, 0:].tolist())

    f = open(dataset.name + ".bin", "wb")
    f.write(dataset.SerializeToString())
    f.close()


def load_features_python(name):
    dataset = features_pb2.Dataset()
    f = open("{0}.bin".format(name), "rb")
    dataset.ParseFromString(f.read())
    f.close()
    X = []
    y = []
    for datum in dataset.data:
        x_i = list(datum.data)
        y_i = datum.label
        X.append(x_i)
        y.append(y_i)

    return np.array(X), np.array(y)

def load_scala_results(name):
    f = open(name, "r")
    result_lines = f.readlines()
    results = np.array(map(lambda x: map(lambda y: float(y), x.split(",")), result_lines))
    labels = results[:, 0]
    weights = results[:, 1:]
    return labels, weights

def load_all_features_from_dir(dirname):
    files = glob.glob(dirname + "/part*")
    all_features = []
    all_labels = []
    i = 1
    for f in files:
        print "Part {0}".format(i)
        features, labels = load_features_from_text(f)
        all_features.extend(features)
        all_labels.extend(labels)
        i += 1
    return np.array(all_features), np.array(all_labels)


def load_features_from_text(fname):
    x_tuples = open(fname).readlines()
    x = map(lambda x: (map(lambda y: float(y), (x[1:-2].split(",")[:-1])), float(x[1:-2].split(",")[-1])), x_tuples)
    features, labels = zip(*x)
    return features, labels


def compute_metrics(exp, y_train, y_train_pred, y_test, y_test_pred):
    exp_flatten = flatten_dict(exp)
    exp_flatten = dict(map(lambda x: (x[0], [str(x[1])]), exp_flatten.items()))
    df = DataFrame.from_dict(exp_flatten)
    train_acc = metrics.accuracy_score(y_train, y_train_pred)
    test_acc = metrics.accuracy_score(y_test, y_test_pred)
    df.insert(len(df.columns), "train_acc", train_acc)
    df.insert(len(df.columns), "test_acc", test_acc)
    return df


def build_ckm(exp, seed):
    layers = exp.get("layers")
    filters = exp.get("filters")
    bandwidth = exp.get("bandwidth")
    patch_sizes = exp.get("patch_sizes")
    verbose = exp.get("verbose")
    def ckm_run(X_train, X_test):
        for i in range(layers):
            patch_shape = (patch_sizes[i], patch_sizes[i])
            X_train, X_test =  ckm_apply(X_train, X_test, patch_shape, bandwidth[i], filters[i], random_state=(seed+i))
        return X_train, X_test
    return ckm_run

def solve(exp, X_train, y_train, X_test, y_test, seed):
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    loss = exp["loss"]
    reg = exp["reg"]
    verbose = exp["verbose"]
    if (loss == "softmax"):
        y_train_pred, y_test_pred = softmax_block_gn(X_train, y_train, X_test, y_test, reg, verbose=True)
    else:
        clf = SGDClassifier(loss=loss, random_state=RANDOM_STATE, alpha=reg, verbose=int(verbose))
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
    return y_train_pred, y_test_pred

def parse_experiment(config_file):
    return load(open(config_file))

if __name__ == "__main__":
    main()
