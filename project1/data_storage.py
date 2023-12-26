import numpy as np

def build_nparray(data):

    features = data[1:,:-1]
    labels = data[1:,-1]

    trainingFeatures = np.array(features).astype('float64')
    trainingLabels = np.array(labels).astype('int')
    
    return(trainingFeatures, trainingLabels)

def build_list(data):

    features = data[1:,:-1]
    labels = data[1:,-1]
        
    trainingFeatures = np.array(features).astype('float64')
    trainingLabels = np.array(labels).astype('int')

    listData = trainingFeatures.tolist()
    listLabels = trainingLabels.tolist()
    
    return(listData, listLabels)

def build_dict(data):
    dictionary = dict()
    labelsDictionary = dict()
    for i in range(len(data) - 1):
        secondDictionary = dict()
        for j in range(len(data[i])):
            if j == len(data[i]) - 1:
                labelsDictionary[i] = int(data[i + 1][j])
            else:
                secondDictionary[data[0][j]] = float(data[i + 1][j])
        dictionary[i] = secondDictionary

    return(dictionary, labelsDictionary)