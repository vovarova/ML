import matplotlib.pyplot as plt
import numpy as np


def loadData(path):
    return np.genfromtxt(path, delimiter=',')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(prediction, Y):
    prediction = np.abs(prediction - 0.000001)
    log_l = (Y) * np.log(prediction) + (1 - Y) * np.log(1 - prediction)
    return log_l.mean()


def logisticFunction(X, theta):
    return sigmoid(X.dot(theta))


def gradient(X, theta, Y):
    h = logisticFunction(X, theta)
    return X.T.dot(Y - h).reshape(theta.shape) / Y.shape[0]


def traineLogisticLodelModel(X, Y, alpha, iterations):
    theta = np.zeros((X.shape[1], 1))

    for i in range(iterations):
        if (i % 50000 == 0):
            costT = cost(logisticFunction(X, theta), Y)
            print("Cost", costT)
        theta += alpha * gradient(X, theta, Y)
    return theta


def run():
    data = loadData("data\\hw2.csv")

    X = np.append(np.ones((data.shape[0], 1)).reshape((data.shape[0], 1)), data[:, :-1], axis=1)
    Y = data[:, -1].reshape(data.shape[0], 1)

    printInitialDataPlot(X, Y)
    # traineSize = round(data.shape[0] * 0.8)
    traineSize = data.shape[0]
    X_training = X[:traineSize, :]
    Y_training = Y[:traineSize]

    theta = traineLogisticLodelModel(X_training, Y_training, 0.001, 900000)
    # theta = np.array([-25.161, 0.206, 0.201]).reshape(3, 1)
    predictionRound = np.round(logisticFunction(X, theta))
    printPredictionResultPlot(X, Y, predictionRound)
    print("Trained theta", theta)
    print("Model accuracy", np.mean(predictionRound == Y_training))


def printInitialDataPlot(X, Y):
    unsuccess = X[Y[:, -1] == 0]
    success = X[Y[:, -1] == 1]
    plt.title("Initial data")
    plt.scatter(unsuccess[:, 1], unsuccess[:, 2], label='First group')
    plt.scatter(success[:, 1], success[:, 2], label='Second group')
    plt.ylabel('X2')
    plt.xlabel('X1')
    plt.legend()
    plt.show()


def printPredictionResultPlot(X_test, Y_test, predictionResult):
    resultOfPred = predictionResult - Y_test
    incorrect = X_test[resultOfPred[:, -1] != 0]
    plt.title("Training result")
    unsuccess = X_test[Y_test[:, -1] == 0]
    success = X_test[Y_test[:, -1] == 1]
    plt.scatter(unsuccess[:, 1], unsuccess[:, 2], label='First group')
    plt.scatter(success[:, 1], success[:, 2], label='Second group')
    plt.scatter(incorrect[:, 1], incorrect[:, 2], label='Incorrect prediction')
    plt.ylabel('X2')
    plt.xlabel('X1')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run()
