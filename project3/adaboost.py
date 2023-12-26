import numpy as np
from sklearn import tree

def adaboost_train(X, Y, max_iter):
    weights = {}
    weights[0] = np.full(len(X), 1/len(X))
    f = []
    alpha = []
    xArray = np.array(X)
    yArray = np.array(Y)

    for count in range(max_iter):
        x2Array = xArray
        y2Array = yArray
        xArray = np.empty(0)
        yArray = np.empty(0)
        weights['%c_'.format(count)] = np.empty(0)
        for e in range(len(weights[count])):
            var = int(weights[count][e]*np.reciprocal(min(weights[count])))
            if var == 0:
                var = 1
            for n in range(var):
                xArray = np.append(xArray, x2Array[e])
                yArray = np.append(yArray, y2Array[e])
                weights['%c_'.format(count)] = np.append(weights['%c_'.format(count)], weights[count][e])
        xArray = np.resize(xArray, [int(xArray.shape[0]/2), 2])
        f.append(tree.DecisionTreeClassifier(max_depth = 1).fit(xArray, yArray)) 
        current_predict = f[count].predict(xArray)

        error_sum = 0
        for i in range(current_predict.shape[0]):
            if yArray[i] != current_predict[i]:
                error_sum += 1
            error = error_sum/len(xArray)
        alpha.append(.5*np.log((1 - error)/error))
        weight_sum = sum(weights[count])

        if count < max_iter - 1:
            weights[count+1] = np.empty(0)
            for i in range(weights['%c_'.format(count)].shape[0]):
                weights[count+1] = np.append(weights[count+1], (1/weight_sum)*weights['%c_'.format(count)][i]*np.exp(-alpha[count]*yArray[i]*current_predict[i]))
    return f, alpha        

def adaboost_test(X, Y, f, alpha):
    accurate_counter = 0
    for sample in range(len(f)):
        accurate_counter += alpha[sample]*f[sample].predict(X)
        current_predict = np.sign(accurate_counter)

    accurate_counter_2 = 0
    for sample in range(len(Y)):
        if current_predict[sample] == Y[sample]:
            accurate_counter_2 += 1
    return accurate_counter_2/len(X) 
