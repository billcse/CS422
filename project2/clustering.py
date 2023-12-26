import numpy as np
from scipy.spatial import distance

def K_Means(X, K, mu):
    muList = []
    counter = 1
    while(counter == 1 or not np.array_equal(mu, muList)):
        for e in range(K):
            cluster = [[],[]]
        
        if(mu == []):
            var1 = np.random.choice(X.shape[0], K, replace = False)
            for i in range(len(var1)):
                mu.append(X[var1[i]])
            mu = np.asarray(mu)
        else:
            mu = mu.astype(np.float32)
            if counter > 1:
                mu = muList
                muList = []

        distances = np.empty(K)
        for x in range(len(X)):
            for i in range(K):
                distances[i] = distance.euclidean(X[x], mu[i])
            minimum = np.argmin(distances)
            cluster[minimum].append(X[x])

        for e in range(K):
            var2 = np.array(cluster[e])
            muList.append(np.mean(var2, axis = 0))
        muList = np.array(muList)
        counter += 1
    return muList
