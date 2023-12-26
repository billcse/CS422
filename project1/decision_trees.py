import numpy as np

# Node class used for trees
class Node:
    def __init__(self, feature=None, nodeThreshold=None, left=None, right=None, leaf=None, grow=None):
        self.feature = feature
        self.nodeThreshold = nodeThreshold
        self.left = left
        self.right = right
        self.leaf = leaf
        self.grow = grow

# Decision tree class used for training
class DecisionTree:
    def __init__(self, maximumDepth=1, minimumS=2):
        self.maximumDepth = maximumDepth
        self.minimumS = minimumS
        self.root = None
        self.grow = None
   
    # This function fits the tree to the size
    def size(self, X,Y):
        self.root = (self.generateTree(X,Y))
        return self.root

    # This function will retrieve prediction
    def prediction(self, sampleN):
        node = self.root
        while node.leaf == None:
            if sampleN[node.feature] <= node.nodeThreshold:
                    node = node.left
            else:
                node = node.right
        return node.leaf

    # This function will get the accuracy after doing a comparison
    def accurateT(self, X, Y):
        measure = []
        for sampleN in X:
            node = self.root
            while node.leaf == None:
                if sampleN[node.feature] <= node.nodeThreshold:
                    node = node.left
                else:
                    node = node.right
            measure.append(node.leaf)
        counter = 0
        for e in range(len(Y)):
            if Y[e] == measure[e]:
                counter += 1
        acc = counter / len(Y)
        return acc, measure
   
    # This function does the actual tree creation
    def generateTree(self, X, Y, depth=0):
        sampleAmount = len(X)
        if depth <= self.maximumDepth and sampleAmount >= self.minimumS:
            greatest = self.splitting(X, Y)
            try:
                if greatest['grow'] > 0:
                    left = self.generateTree(X=greatest['negativeX'], Y=greatest['negativeY'], depth=depth + 1)
                    right = self.generateTree(X=greatest['positiveX'], Y=greatest['positiveY'], depth=depth + 1)
                    return Node(feature=greatest['featureI'], nodeThreshold=greatest['nodeThreshold'], left=left, right=right, grow=greatest['grow'])
            except KeyError:
                pass
        var1 = np.count_nonzero(Y == 1)
        var2 = len(Y) - var1
        if var1 > var2:
            return Node(leaf=1)
        elif var2 > var1:
            return Node(leaf=0)
        else:
            return Node(leaf=np.random.randint(1))

    # This function will calculate the entropy
    def calculateEntropy(self, sampleN):
        pick = np.bincount(np.array(sampleN, dtype=np.int64))
        probability = pick / len(sampleN)
        entropy = -np.sum([p*np.log(p) for p in probability if p > 0])
        return entropy

    # This function will calculate the information gain
    def informationGain(self, parentNode, left, right):
        leftVariable = len(left) / len(parentNode)
        rightVariable = len(right) / len(parentNode)
        return self.calculateEntropy(parentNode) - leftVariable * self.calculateEntropy(left) - rightVariable * self.calculateEntropy(right)

    # This function will find the best split
    def splitting(self, X, Y):
        greatest = {}
        bestinformationGain = -1
        amountOfFeatures = len(X[0])
        for f in range(amountOfFeatures):
            featureN = X[:,f]
            for nodeThreshold in np.unique(featureN):
                negativeX = []
                negativeY = []
                positiveX = []
                positiveY = []
                for i in range(len(featureN)):
                    if featureN[i] <= nodeThreshold:
                        negativeX.append(X[i])
                        negativeY.append(Y[i])
                    else:
                        positiveX.append(X[i])
                        positiveY.append(Y[i])
                negativeX = np.array(negativeX)
                negativeY = np.array(negativeY)
                positiveX = np.array(positiveX)
                positiveY = np.array(positiveY)
                if len(negativeX) > 0 and len(positiveX) > 0:
                    grow = self.informationGain(Y, negativeY, positiveY)
                    if grow > bestinformationGain:
                        bestinformationGain = grow
                        greatest = {
                            'featureI': f,
                            'nodeThreshold': nodeThreshold,
                            'negativeX': negativeX,
                            'negativeY': negativeY,
                            'positiveX': positiveX,
                            'positiveY': positiveY,
                            'grow': grow
                        }
        return greatest
    
# This function trains the tree, as required in instructions
def DT_train_binary(X,Y,max_depth):
    tree = DecisionTree(max_depth)
    tree.size(X,Y)
    return tree
   
# This function tests the tree, as required in instructions
def DT_test_binary(X,Y,tree):
    return tree.accurateT(X, Y)
   
# This function makes a prediction, as required in instructions
def DT_make_prediction(X,tree):
    return tree.prediction(X)

# This class makes a forest consisting of decision trees
class RandomForest():
    def __init__(self, numTrees=1, maximumDepth=1):
        self.numTrees = numTrees
        self.maximumDepth = maximumDepth
        self.trees = []
        
    # This function creates the forest of trees
    def generateForest(self,X,Y):
        for i in range(self.numTrees):
            tree = DecisionTree(self.maximumDepth)
            y = []
            x = []
            for j in range(30):
                position = np.random.randint(len(X))
                x.append(X[position])
                y.append(Y[position])
            x = np.array(x)
            y = np.array(y)
            tree.size(x,y)
            self.trees.append(tree)
      
    # This function will find accuracy of tree by traversing through it
    def accurateT(self,X, Y):
        measure=[]
        average=[]
        treeAccuracy = []
        for e in range(len(self.trees)):
            measure.append([])
            currentAccuracy, measure[e] = self.trees[e].accurateT(X,Y)
            treeAccuracy.append(currentAccuracy)
        measure = np.array(measure)

        for e in range(len(measure[0])):
            var1 = np.count_nonzero(measure[:,e] == 1)
            var2 = self.numTrees - var1
            if var1 > var2:
                average.append(1)
            elif var2 > var1:
                average.append(0)
            else:
                average.append(np.random.randint(1))
        counter = 0
        for e in range(len(average)):
            if Y[e] == average[e]:
                counter += 1
        acc = counter / len(average)

        for e in range(len(treeAccuracy)):
            print('DT %d: ' %e, treeAccuracy[e])
        return acc
  
# This function builds the forest
def RF_build_random_forest(X,Y,max_depth,num_of_trees):
    forest = RandomForest(num_of_trees, max_depth)
    forest.generateForest(X,Y)
    return forest
  
# This function tests the forest
def RF_test_random_forest(X,Y,RF):
    return RF.accurateT(X,Y)

