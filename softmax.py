import numpy as np
from sklearn import metrics

def softmax(X_train, y_train, X_test, y_test, multiplier=1e-2, numiter=50):
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
        return y_train_pred, y_test_pred

