import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
# Simple SVM for classification

def generateSample(N, variance=100, m=0):
    X = np.matrix(range(N)).T + 1
    Y = np.matrix([random.random() * variance + i * 10 + 900 for i in range(len(X))]).T + m

    return X, Y


def fitModel_gradient(x1,x2,y1,y2):
    N = len(x1)
    w = np.zeros((x1.shape[1], 1))
    a = 0.0001

    maxIteration = 95000
    for i in range(maxIteration):
        error1 = x1 * w - y1
        error2 = x2 * w - y2
        gradient1 = x1.T * error1 / N
        gradient2 = x1.T * error2 / N
        w = w - a * (gradient1 + gradient2)

    line_for_class_1 = x1 * w - y1
    line_for_class_2 = x2 * w - y2
    cosine = np.max(x1) / w[0]

    # for support vector

    class_1_idx = np.argmin(abs(line_for_class_1 * cosine))
    class_2_idx = np.argmin(abs(line_for_class_2 * cosine))


    plt.scatter(x1[class_1_idx, 1],y1[class_1_idx, 0])
    plt.plot(x1[:,1],x1*w  + y1[class_1_idx, 0] - x1[class_1_idx]*w)
    plt.scatter(x2[class_2_idx,1], y2[class_2_idx, 0])
    plt.plot(x2[:,1],x2*w + y2[class_2_idx, 0] - x2[class_2_idx]*w)

    plt.plot(x1[:,1],x1*w)

    '''
    def Loss_SVM(margin1, margin2):
        #return abs(1 / margin1) + abs(1 / margin2)
        return np.log(abs(margin1)) + np.log(abs(margin2))



    #margin considering

    cosine = np.max(x1) / w[0]

    line_for_class_1 = x1 * w - y1
    line_for_class_2 = x2 * w - y2
    margin_for_class_1 = np.min(line_for_class_1 * cosine)
    margin_for_class_2 = np.min(line_for_class_2 * cosine)

    while(True):
        line_for_class_1 = x1 * w - y1
        line_for_class_2 = x2 * w - y2
        margin_for_class_1 = np.min(line_for_class_1 * cosine)
        margin_for_class_2 = np.min(line_for_class_2 * cosine)
        Loss = Loss_SVM(margin_for_class_1,margin_for_class_2)
        
        

        w = w - [[0], [0.1]]
        line_for_class_1 = x1 * w - y1
        line_for_class_2 = x2 * w - y2
        margin_for_class_1_perturbed = np.min(line_for_class_1 * cosine)
        margin_for_class_2_perturbed = np.min(line_for_class_2 * cosine)
        Loss_SVM_perturbed_bias_direction = Loss_SVM(margin_for_class_1_perturbed, margin_for_class_2_perturbed)
        w = w - [[0],[Loss_SVM_perturbed_bias_direction - Loss]]

        if Loss > Loss_SVM_perturbed_bias_direction:
            w = w - [[0], [Loss_SVM_perturbed_bias_direction - Loss]]
            print("ORIGIN LOSS" , Loss)
            print("BIAS DIRECTION MOVED" , Loss_SVM_perturbed_bias_direction)

        else:

            return w
    '''
    return w






def plotModel(x1,x2,y1,y2,w):
    plt.plot(x1[:,1], y1, "x")
    plt.plot(x2[:,1], y2, "x")
    x = np.concatenate((x1,x2))
    plt.plot(x[:,1], x * w, "r-")

    plt.show()

def test(N, variance, modelFunction):
    X1, Y1 = generateSample(N, variance)
    X2, Y2 = generateSample(N, variance, 1000)
    X1 = np.hstack([np.matrix(np.ones(len(X1))).T, X1])
    X2 = np.hstack([np.matrix(np.ones(len(X2))).T, X2])
    w = modelFunction(X1,X2,Y1,Y2)
    plotModel(X1,X2,Y1,Y2,w)


test(100, 500, fitModel_gradient)