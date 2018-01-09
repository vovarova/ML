import numpy as np
from sklearn.datasets import fetch_mldata


def loadData():
    mnist = fetch_mldata('MNIST original', data_home='data\\')
    x = np.array(mnist.data, dtype=float) / 255
    y = mnist.target
    return x, y


def indicatorMatrix(y, groups):
    m = y.shape[0]
    new_y = np.zeros((m, groups))
    for idx, val_y in np.ndenumerate(y):
        new_y[idx, int(y[idx])] = 1
    return new_y


def run():
    x, y = loadData()
    train_x = x[0:60000]
    train_y = y[0:60000]
    trained_w = traineSoftmaxModel(train_x, train_y, 1, 2000)
    print("Model accuracy", np.mean(getPredictedNumber(x[60000:-1], trained_w) == y[60000:-1]))
    pass


def getPredictedNumber(x, w):
    return np.argmax(softMax(x, w), axis=1)


def loadModel():
    return np.loadtxt('data\\soft-max.txt')


def saveModel(w):
    np.savetxt('data\\soft-max.txt', w)


def softMax(x, w):
    score = np.dot(x, w)
    exp_score = np.exp(score)
    sm = exp_score / np.sum(exp_score, axis=1).reshape(exp_score.shape[0], 1)
    return sm


def getLossAndGrad(w, x, y_indicator):
    m = x.shape[0]
    prob = softMax(x, w)
    log_prob = np.log(np.abs(prob - 0.0000001))
    loss = (-1 / m) * np.sum(y_indicator * log_prob)
    grad = (-1 / m) * x.T.dot(y_indicator - prob).reshape(w.shape)
    return loss, grad


def traineSoftmaxModel(x, y, alpha, iterations):
    groups = int(np.max(y) - np.min(y) + 1)
    w = loadModel()
    if w.shape[0] == 0:
        w = np.zeros((x.shape[1], groups))
    y_indicator = indicatorMatrix(y, groups)
    min_loss = 0.273251372543
    for i in range(iterations):
        loss, grad = getLossAndGrad(w, x, y_indicator)
        if (i % 30 == 0):
            print("Loss", loss)
            if (loss < min_loss):
                min_loss = loss
                saveModel(w)
            print("Iteration", i)
        w = w - alpha * grad

    saveModel(w)
    return w


if __name__ == '__main__':
    run()
