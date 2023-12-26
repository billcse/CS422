import numpy as np

def perceptron_train(X,Y):
    w1 = 0
    w2 = 0
    b = 0

    refresh = True
    while refresh == True:
        tempVar = 0
        for e in range(len(X)):
            if((X[e][0]*w1) + (X[e][1]*w2) + b)* Y[e] <= 0:
                w1 = w1 + X[e][0]*Y[e]
                w2 = w2 + X[e][1]*Y[e]
                b = b + Y[e]
            else:
                tempVar += 1
        if tempVar == len(X):
            refresh = False
    weights = [[w1, w2], b]

    print(weights)
    return weights

def perceptron_test(X_test, Y_test, w, b):
    sum = np.sum(X_test*w, axis = 1) + b
    y = np.where(sum > 0, 1, -1)
    return (y == Y_test).sum()/len(Y_test)