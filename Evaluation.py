import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix



def countMisclassifications(network, testData, testTarget):
    errorCount = 0
    for i in range(len(testData)):
        t = np.argmax(np.asarray(testTarget[i]))
        o = np.argmax(np.asarray(network.GetOutput(testData[i])))
        
        if t != o:
            errorCount += 1

    return errorCount

def MultiClassToOne(network, testData, testTarget):
    t =[]
    o =[]
    for i in range(len(testData)):
        t.append(np.argmax(np.asarray(testTarget[i])))
        o.append(np.argmax(np.asarray(network.GetOutput(testData[i]))))
    return t,o

def GetPredictions(network, testData, testTarget):
    t =[]
    o =[]
    for i in range(len(testData)):
        t.append(testTarget[i])
        o.append(network.GetOutput(testData[i]))
    return t,o

def PrintMetrics(title, traget, output, time):
    print(">>>>>>>>>>>>> {}".format(title))
    print("F1            : {}".format(round(f1_score(traget, output, average="macro"),5)))
    print("Precision     : {}".format(round(precision_score(traget, output, average="macro"),5)))
    print("Recall        : {}".format(round(recall_score(traget, output, average="macro"),5)))
    print("Training Time : {} Seconds".format(round(time,3)))
        
    