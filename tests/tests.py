from experiments import *
from nose.plugins.attrib import attr

from tabulate import tabulate

@attr("implemented")
def python_mnist_features_test():
    ''' Generate features on a tiny subset of MNIST
        from a 1 layer CKN
        check if mean of features is reasonable
    '''
    exp = load(open("./tests/mnist_small.exp"))
    dataset = exp["dataset"]
    seed = exp["seed"]
    (X_train, y_train), (X_test, y_test) = load_data(dataset)
    X_train_lift, X_test_lift = gen_features(exp, X_train, X_test, seed)
    assert 0.6 <= np.mean(np.mean(X_train_lift)) <= 0.8

@attr("implemented")
def python_mnist_save_load_test():
    ''' Saves/Loads CKM features + labels to disk to see if
        protobuff serialization works
    '''
    exp = load(open("./tests/mnist_small.exp"))
    dataset = exp["dataset"]
    seed = exp["seed"]
    (X_train, y_train), (X_test, y_test) = load_data(dataset)
    X_train_lift, X_test_lift = gen_features(exp, X_train, X_test, seed)
    save_features_python(X_train_lift, y_train, "x_test")
    X_load, y_load = load_features_python("x_test")
    assert np.all(y_load == y_train)
    assert np.all(X_load == X_train_lift)


@attr("implemented")
def python_mnist_sanity_test():
    ''' Runs a 1 layer ckn on a tiny subset of MNIST
        should get 97.6 percent on test set '''
    results = python_run(load(open("./tests/mnist_small.exp")))
    print ""
    print tabulate(results, headers="keys")
    assert(float(results['test_acc'][0]) >= 0.9805)


@attr('slow', 'implemented')
def python_mnist_full_test():
    ''' Runs a 2 layer ckn on all of MNIST
        should get 99.4 percent on test set '''
    results = python_run(load(open("./tests/mnist_full.exp")))
    dataset 
    assert(float(results['test_acc'][0]) >= 0.994)

@attr('slow', 'implemented', 'scala')
def scala_mnist_sanity_test():
    ''' Runs a 2 layer ckn via keystone on a tiny susbet of MNIST
        should get 99.64 percent on test set '''
    results = scala_run(load(open("./tests/sample_scala.exp")))
    assert(float(results['test_acc'][0]) >= 0.996)


@attr('slow', 'scala')
def scala_mnist_full_test():
    ''' Runs a 2 layer ckn via keystone on all of MNIST
        should get 99.46 percent on test set '''
    results = scala_run(load(open("./tests/sample_scala.exp")))
    assert(float(results['test_acc'][0]) >= 0.994)
