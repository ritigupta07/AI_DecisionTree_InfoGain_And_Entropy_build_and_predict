import pandas as pd
import operator
import math
import numpy as np
import os

class DecisionStump:
    def __init__(self):
        self.root = None

    def fit(self, df):
        attributes = list(df)[0:len(df.columns) - 1]
        sample = df.iloc[0:]
        labels = df.iloc[:, -1]
        decisionTree = DecisionTree()
        self.root = decisionTree.id3(sample, attributes, labels)

    def predict(self, df):
        predList = []
        for index, a in df.iterrows():
            predList.append(self.root.childs[a[self.root.value]].value)
        return np.array(predList)


class Node:
    def __init__(self):
        self.value = None
        self.childs = {}

class DecisionTree:

    def isSingleLabeled(self, labels):
        label = labels[0]
        for l in labels:
            if l != label:
                return False
        return True

    def getLabelsCount(self, labels):
        labelsCount = {}
        for label in labels:
            if label in labelsCount:
                labelsCount[label] += 1
            else:
                labelsCount[label] = 1
        return labelsCount

    def getDominantLabel(self, labels):
        return max(self.getLabelsCount(labels).items(), key=operator.itemgetter(1))[0]

    def getEntropy(self,labels):
        labelsCount = self.getLabelsCount(labels)
        totalLabels = len(labels)

        entropy = 0
        for key, value in labelsCount.items():
            fraction = value/totalLabels
            entropy = entropy - (fraction * math.log(fraction,2))
        return entropy

    def getInformationGain(self, attribute, samples, labels):
        attrVal = {}
        rootEntropy = self.getEntropy(labels)
        infoGain = rootEntropy
        for index, a in samples.iterrows():
            if a[attribute] not in attrVal:
                attrVal[a[attribute]] = []
            attrVal[a[attribute]].append(a[list(samples)[-1]])

        for key, value in attrVal.items():
            childEntropy = self.getEntropy(value)
            infoGain = infoGain - ((len(value)/len(labels))*childEntropy)

        print(str(attribute) + " : " + str(infoGain))
        return infoGain

    def getBestAttribute(self, attributes, samples, labels):
        attrInformationGain = {}
        print("Information Gain : ")
        for attr in attributes:
            attrInformationGain[attr] = self.getInformationGain(attr,samples, labels)
        return max(attrInformationGain.items(), key=operator.itemgetter(1))[0]

    def id3(self,samples, attributes, labels):
        root = Node()

        if self.isSingleLabeled(labels):
            root.value = labels[0]
            return root

        if len(attributes) == 0:
            root.value = self.getDominantLabel(labels)
            return root

        bestAttr = self.getBestAttribute(attributes, samples,labels)
        root.value = bestAttr

        attrVal = []
        for index, a in samples.iterrows():
            if a[bestAttr] not in attrVal:
                attrVal.append(a[bestAttr])

        for val in attrVal:
            root.childs[val] = Node()
            leafLabels = []
            for index, a in samples.iterrows():
                if a[bestAttr] == val:
                    leafLabels.append(labels[index])
            root.childs[val].value = self.getDominantLabel(leafLabels)
        return root


def test():
    train_file = './train_data.csv'
    test_file = './test_data.csv'

    ds = DecisionStump()

    if os.stat(train_file).st_size != 0:
        df = pd.read_csv(train_file)
        if not df.empty:
            ds.fit(df)

    if os.stat(test_file).st_size != 0:
        df_test = pd.read_csv(test_file)
        if not df_test.empty and ds.root != None:
            pred_val = ds.predict(df_test)
            print("\nPredictions : ")
            print(pred_val)

if __name__ == '__main__':
    test()
