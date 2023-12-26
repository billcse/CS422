import numpy as np
from scipy.spatial import distance

def KNN_test(X_train,Y_train,X_test,Y_test,K):
    xTrain = len(X_train)
    xTest = len(X_test)
    distances = np.zeros(xTrain)
    
    accuracyPrediction = 0
    for i in range(xTest):
        for j in range(xTrain):
            distances[j] = distance.euclidean(X_train[i],X_test[j])
            
        iTrain = np.argsort(distances)
        result = 0
        for k in range(K):
            result = result +  Y_train[iTrain[k]]
            print((distances[iTrain[k]], "Y:", Y_train[iTrain[k]] , "Result:", result)) 
    
        if result > 0:
            y = 1
        else:
            y = -1
        if Y_test[i] == y:
            accuracyPrediction += 1            
    accuracy = accuracyPrediction / xTest
    return accuracy
