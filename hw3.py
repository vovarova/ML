import numpy as np
from sklearn.datasets import fetch_mldata


def loadData():
    mnist = fetch_mldata('MNIST original', data_home='data\\')
    x = mnist.data
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
    #trained_w = traineSoftmaxModel(x, y, 0.00001, 2000)

    print("Model accuracy", np.mean(getPredictedNumber(x, loadModel()) == y))
    # w = loadModel()
    # i = 10000
    # soft_max = softMax(x[i].reshape(1, x.shape[1]), w)
    # preds = np.argmax(soft_max, axis=1)

    # print(preds)
    # print(y[i])
    pass


def getPredictedNumber(x, w):
    return np.argmax(softMax(x, w), axis=1)


def loadModel():
    return np.loadtxt('data\\soft-max.txt')


def saveModel(w):
    np.savetxt('data\\soft-max.txt', w)


def softMax(x, w):
    score = np.dot(x, w)
    # score -= np.max(score)
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
    w = loadModel()
    groups = int(np.max(y) - np.min(y) + 1)
    y_indicator = indicatorMatrix(y, groups)
    for i in range(iterations):
        loss, grad = getLossAndGrad(w, x, y_indicator)
        if (i % 30 == 0):
            print("Loss", loss)
            saveModel(w)
            print("Iteration", i)
        w = w - alpha * grad

    saveModel(w)
    return w


if __name__ == '__main__':
    run()
