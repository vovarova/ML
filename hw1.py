import os

import numpy as np


# Функція передбачає ціну оренди квартири, використовуючи лінійну регресію
# з ознаками (фічами) X та вагами theta
def getLinear(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.dot(x, theta.transpose()).reshape(x.shape[0], 1)


# Функція рахує помилку передбачення, якщо передбачені значення - h,
# а реальні значення - y
def getError(prediction: np.ndarray, y: np.ndarray) -> np.ndarray:
    simpleError = prediction - y
    return simpleError * simpleError / 2


# Функція рахує градієнт (часткові похідні) помилки, які необхідні
# для здійснення "кроку" навчання алгоритму
def getGradient(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # print(x)
    return (getLinear(theta, x) - y).transpose().dot(x)


# Функція робить один "крок" градієнтного спуску і повертає значення
# оновлених вагів next_theta
def getGradientDescentStep(theta: np.ndarray, alpha, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return theta - alpha * getGradient(theta, x, y)


# Функція розраховує значення вагів за допомогою нормальних рівнянь
def getNormalEquations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y).reshape((1, x.shape[1]))


# Функція передбачає значення y в точці x_query за допомогою зваженої лінійної регресії (weighted linear regression).
# Для цього необхідно зробити iter кроків градієнтного спуску для навчання зваженої лінійної регресії, а після
# цього використати навчені ваги для передбачення y.
def getWeightedLRPrediction(alpha, iter, x_query: np.ndarray, tau, x: np.ndarray, y: np.ndarray):
    weights = calculateWeights(x, x_query, tau)

    theta = np.zeros((1, x.shape[1]))

    e = 0.001
    oldCost = 0
    newCost = oldCost + e + 1
    i = 0
    while abs(oldCost - newCost) > e and i < iter:
        oldCost = sum(weights * getError(getLinear(theta, x), y))
        theta = getWeightedGradientDescentStep(theta, alpha, x, y, weights)
        newCost = sum(weights * getError(getLinear(theta, x), y))
        i += 1

    diagonalWeight = np.diag(weights.flat)
    bestTheta = np.linalg.inv(x.transpose().dot(diagonalWeight).dot(x)).dot(x.transpose()).dot(diagonalWeight).dot(
        y).reshape((1, x.shape[1]))

    print("Not best theth ",getLinear(theta,x_query)*1000)
    #print(np.linalg.inv(x.transpose().dot(diagonalWeight).dot(x)).dot(x.transpose()).dot(diagonalWeight).dot(y).reshape((1, x.shape[1])))
    return getLinear(bestTheta,x_query)


def calculateWeights(x, x_query, tau) -> np.ndarray:
    return np.exp(np.sum(-(x - x_query) ** 2, axis=1) / (2 * tau)).reshape(x.shape[0], 1)


def getWeightedGradient(theta: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (w * (getLinear(theta, x) - y)).transpose().dot(x)


def getWeightedGradientDescentStep(theta: np.ndarray, alpha, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return theta - alpha * getWeightedGradient(theta, x, y, w)


def getGradientDescentStep(theta: np.ndarray, alpha, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return theta - alpha * getGradient(theta, x, y)


def getWeightedLRGradient(x, ):
    pass


def run():
    # Вкажіть тут свій шлях до папки, що містить файл із даними
    base_path = "E:\\Study\\lits\\ml1\\programming"
    filename = "prices.csv"
    path_to_file = os.path.join(base_path, filename)
    data = np.genfromtxt(path_to_file, delimiter=',')

    X = data[1:, 0:8]
    X = np.dot(X, np.linalg.inv(np.diag(np.linalg.norm(X, axis=0))))
    X_training = X[0:78, 0:8]
    X_test = X[78:, 0:8]

    Y_training = np.array([data[1:79, 8] / 1000]).T
    Y_test = np.array([data[79:, 8] / 1000]).T
    theta = np.zeros((X_training.shape[1])).reshape((1, X_training.shape[1]))
    prediction = getLinear(theta, X_training)

    # -------------- Testing linear regression prediction --------------

    print("------------------------------------------------------------")
    print("Testing getLinear...")
    if prediction.shape[0] == 78 and sum(prediction) == 0:
        print("correct")
    else:
        print("incorrect")

    # -------------- Testing error function --------------

    print("------------------------------------------------------------")
    print("Testing getError...")
    error = getError(prediction, Y_training)
    if (error.shape[0] == 78 and error[0, 0] == 40.5 and error[2, 0] == 2.53125):
        print("correct")
    else:
        print("incorrect")

    # -------------- Testing gradient function --------------

    print("------------------------------------------------------------")
    print("Testing getGradient...")
    gradient = getGradient(theta, X_training, Y_training)
    if (gradient.shape[1] == 8 and round(gradient[0][0] + 32.51) == 0 and round(gradient[0][2] + 11.05) == 0):
        print("correct")
    else:
        print("incorrect")

    # -------------- Testing gradient descent --------------

    print("Testing gradient descent...")

    oldCost = 0
    newCost = 10
    i = 0
    while abs(oldCost - newCost) > 0.01 and i < 9000:
        oldCost = np.sum(getError(getLinear(theta, X_training), Y_training))
        i += 1
        theta = getGradientDescentStep(theta, 0.02, X_training, Y_training)
        newCost = np.sum(getError(getLinear(theta, X_training), Y_training))

        if np.mod(i, 1000) == 0:
            print("Iteration %d, sum error = %d", i, sum(getError(getLinear(theta, X_training), Y_training)))

    print("------------------------------------------------------------")
    print("Testing getGradientDescentStep...")
    if (round(theta[0][0]) == 13 and round(theta[0][7]) == -29):
        print("correct")
    else:
        print("incorrect")

    print("Thetas leasred by gradient descent:")
    trained_theta = theta

    print("------------------------------------------------------------")
    print("Testing getNormalEquations...")
    best_theta = getNormalEquations(X_training, Y_training)
    if (round(best_theta[0][0] - 12.11628) == 0 and round(best_theta[0][2] - 0.54360) == 0):
        print("correct")
    else:
        print("incorrect")

    print("Thetas derived by normal equations:")
    print(best_theta)

    print("------------------------------------------------------------")
    # x_query = X_test[test_example_id, :].reshape((1, -1))
    for i in range(X_test.shape[0]):
        x_query = X_test[i, :].reshape((1, X_test.shape[1]))
        predicted_wlr = getWeightedLRPrediction(0.02, 9000, x_query, 0.05, X_training, Y_training) * 1000
        predicted_lr = getLinear(trained_theta, x_query) * 1000

        print("Predicted price with weighted linear regression = " + str(predicted_wlr))
        print("Predicted price with standard linear regression = " + str(predicted_lr))
        print("Real price = " + str(Y_test[i] * 1000))
        print("Testing getWeightedLRPrediction...")
        if abs(predicted_lr - Y_test[i] * 1000) > abs(predicted_wlr - Y_test[i] * 1000):
            print("correct")
        else:
            print("incorrect")


if __name__ == '__main__':
    run()
