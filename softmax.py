import numpy as np
from sklearn import metrics
import math

def backtrack(obj, grad_fn, x, beta=0.8, gamma=0.4, t=1):
    armigo = False
    i = 0
    while (not armigo):
        i += 1
        direction = -grad_fn(x)
        armigo =  obj(x + t*direction ) <= obj(x) + gamma*t*(direction.dot(grad_fn(x)))
        t = t*beta
    return t

def softmax_gn(X_train, y_train, X_test, y_test, multiplier=1e-2, numiter=50, verbose=False):
        ''' Implementation of gauss-newton quassi-newton optimization algorithm
            with softmax objective
        '''
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        gmat = (1.0/X_train.shape[0])*(X_train.T.dot(X_train))
        lambdav = multiplier*np.trace(gmat)/gmat.shape[0]
        gmat = gmat + lambdav*np.eye(gmat.shape[0]);
        w = np.zeros((10, gmat.shape[0]))
        num_samples = X_train.shape[0]
        onehot = lambda x: np.eye(10)[x]
        y_train_hot = np.array(map(onehot, y_train))
        y_test_hot = np.array(map(onehot, y_test))
        for k in range(numiter):
            train_preds  = w.dot(X_train.T).T # 60000 x 10
            train_preds = train_preds - np.max(train_preds, axis=1)[:,np.newaxis]
            train_preds = np.exp(train_preds)
            train_preds = train_preds/(np.sum(train_preds, axis=1)[:,np.newaxis])
            train_preds = y_train_hot - train_preds
            grad = (1.0/num_samples)*(X_train.T.dot(train_preds).T) - lambdav*w
            w = w + (np.linalg.solve(gmat, grad.T)).T
            y_train_pred = np.argmax(w.dot(X_train.T).T, axis=1)
            y_test_pred = np.argmax(w.dot(X_test.T).T, axis=1)
            train_acc = metrics.accuracy_score(y_train, y_train_pred)
            test_acc = metrics.accuracy_score(y_test, y_test_pred)
            if (verbose):
              print "Iter: {0}, Train Accuracy: {1}, Test Accuracy: {2}".format(k, train_acc, test_acc)
        return y_train_pred, y_test_pred, w

def softmax_gd(X_train, y_train, X_test, y_test, multiplier=1e-2, numiter=100, verbose=False, step=lambda x: 0.0001):
        ''' Implementation of gauss-newton quassi-newton optimization algorithm
            with softmax objective
        '''
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        #gmat = (1.0/X_train.shape[0])*(X_train.T.dot(X_train))
        #lambdav = multiplier * np.mean(X_train * X_train)
        lambdav = multiplier
        #gmat = gmat + lambdav*np.eye(gmat.shape[0]);
        w = np.zeros((10, X_train.shape[1]))
        num_samples = X_train.shape[0]
        onehot = lambda x: np.eye(10)[x]
        y_train_hot = np.array(map(onehot, y_train))
        y_test_hot = np.array(map(onehot, y_test))
        loss = 0
        for k in range(numiter):
            print("Iteration " + str(k))
            train_preds  = w.dot(X_train.T).T # 60000 x 10
            train_preds = train_preds - np.max(train_preds, axis=1)[:,np.newaxis]
            train_preds = np.exp(train_preds)
            train_preds = train_preds/(np.sum(train_preds, axis=1)[:,np.newaxis])
            train_preds_ = y_train_hot - train_preds
            grad = (1.0/num_samples)*(X_train.T.dot(train_preds_).T) - lambdav*w
            #w = w + (np.linalg.solve(gmat, grad.T)).T
            w_0 = w
            w = w  + step(k)*grad
            print train_preds.shape
            print y_train_hot.shape
            loss_0 = loss
            loss = - np.sum(y_train_hot * np.log(train_preds + 1e-12))
            print "Loss ", loss

            print "Loss Change", loss - loss_0
            print "grad magnitude", np.linalg.norm(grad)
            y_train_pred = np.argmax(w.dot(X_train.T).T, axis=1)
            y_test_pred = np.argmax(w.dot(X_test.T).T, axis=1)
            train_acc = metrics.accuracy_score(y_train, y_train_pred)
            test_acc = metrics.accuracy_score(y_test, y_test_pred)
            if (verbose):
              print "Iter: {0}, Train Accuracy: {1}, Test Accuracy: {2}".format(k, train_acc, test_acc)
        return y_train_pred, y_test_pred

def softmax_block_gn(X_train, y_train, X_test, y_test, multiplier=1e-2, numiter=10,block_size=4000, epochs=1, verbose=False):
        ''' Fix some coordinates '''
        total_features = X_train.shape[1]
        num_blocks = math.ceil(total_features/block_size)
        w = np.zeros((10, X_train.shape[1]))
        num_samples = X_train.shape[0]
        onehot = lambda x: np.eye(10)[x]
        y_train_hot = np.array(map(onehot, y_train))
        y_test_hot = np.array(map(onehot, y_test))
        lambdav = multiplier*np.mean(X_train * X_train)
        loss = 0
        print num_blocks
        for e in range(epochs):
                shuffled_features = np.random.choice(total_features, total_features, replace=False)
                for b in range(int(num_blocks)):
                        block_features = shuffled_features[b*block_size:min((b+1)*block_size, total_features)]
                        X_train_block = X_train[:, block_features]
                        X_test_block = X_test[:, block_features]
                        w_block = w[:, block_features]
                        gmat = (1.0/X_train_block.shape[0])*(X_train_block.T.dot(X_train_block))
                        gmat = gmat + lambdav*np.eye(gmat.shape[0]);
                        w_full = np.zeros((10, X_train.shape[1]))
                        for k in range(numiter):
                            print "Newton iter ", k
                            train_preds  = w.dot(X_train.T).T # datapoints x 10
                            train_preds = train_preds - np.max(train_preds, axis=1)[:,np.newaxis]
                            train_preds = np.exp(train_preds)
                            train_preds = train_preds/(np.sum(train_preds, axis=1)[:,np.newaxis])
                            train_preds = y_train_hot - train_preds
                            grad = (1.0/num_samples)*(X_train_block.T.dot(train_preds).T) - lambdav*w_block # blocksize x 1
                            w_block = w_block + (np.linalg.solve(gmat, grad.T)).T
                            w[:, block_features] = w_block
                        loss = - np.sum(y_train_hot * np.log(train_preds + 1e-10))
                        y_train_pred = np.argmax(w.dot(X_train.T).T, axis=1)
                        y_test_pred = np.argmax(w.dot(X_test.T).T, axis=1)
                        train_acc = metrics.accuracy_score(y_train, y_train_pred)
                        test_acc = metrics.accuracy_score(y_test, y_test_pred)
                        if (verbose):
                                print "Epoch: {0}, Block: {3}, Loss: {4}, Train Accuracy: {1}, Test Accuracy: {2}".format(e, train_acc, test_acc, b, loss)


