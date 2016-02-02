import argparse

from yaml import load, dump

from pandas import DataFrame
from ckm import *

import logging
import collections
import os
import subprocess

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
    X_train_lift, X_test_lift = gen_features(exp, X_train, X_test, seed)
    y_train_pred, y_test_pred = solve(exp, X_train_lift, y_train, X_test_lift, y_test, seed)
    results = compute_metrics(exp, y_train, y_train_pred, y_test, y_test_pred)
    return results

def scala_run(exp):
    expid = exp["expid"]
    config_yaml = "./tests/sample_scala.exp"
    env = os.environ.copy()
    env['KEYSTONE_MEM'] = str('32g')
    env['SPARK_EXECUTOR_CORES'] = str(32)
    env['OMP_NUM_THREADS'] = str(1)
    # sanity check before running the process

    # if not os.path.isdir(outdir):
    #     raise ValueError("output dir must exist")

    logfile = expid + ".spark.log"
    # if os.path.exists(logfile) and output_sanity_check:
    #     raise ValueError("output dir has logfile, should be empty")
    pipelineClass="pipelines.CKM"
    pipelineJar = "/work/vaishaal/ckm/keystone_pipeline/target/scala-2.10/ckm-assembly-0.1.jar"
    if not os.path.exists(pipelineJar):
        raise ValueError("Cannot find pipeline jar")

    # basically capturing the popen output and saving it to disk  and
    # printing it to screen are a multithreaded nightmare, so we let
    # tee do the work for us

    ## pipefail set so that we get the correct process return code
    cmd = " ".join(["./keystone_pipeline/bin/run-pipeline.sh", pipelineClass,
                                   pipelineJar, config_yaml])

    print cmd
    p = subprocess.Popen(" ".join(["./keystone_pipeline/bin/run-pipeline.sh", pipelineClass,
                                   pipelineJar, config_yaml]),
                         shell=True, executable='/bin/bash')
    #p = subprocess.Popen(cmd, shell=True, executable='/bin/bash', env = env)
    p.wait()

    if p.returncode != 0:
        raise Exception("invocation terminated with non-zero exit status")

def gen_features(exp, X_train, X_test, seed):
    ckm_run = build_ckm(exp, seed)
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


def build_ckm(exp, seed):
    layers = exp.get("layers")
    filters = exp.get("filters")
    bandwidth = exp.get("bandwidth")
    patch_sizes = exp.get("patch_sizes")
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
