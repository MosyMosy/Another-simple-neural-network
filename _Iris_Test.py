import Network as net
import ActivationFunction as af
import Evaluation as ev
import numpy as np
import os
import math
import time

_DataSetPath = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'DataSet')   # Get the directory for StringFunctions
_curentdatapath = os.path.join(_DataSetPath, "iris-oneVall.data")

targetsEnum = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

data = np.loadtxt(_curentdatapath, dtype=float, delimiter=",")
np.random.shuffle(data)

for i in range(len(data[0,:-3])):
    _max = np.max(data[:,i])
    _min = np.min(data[:,i])
    data[:,i] = (data[:,i]-_min)/(_max - _min)

trainPortion = int(len(data)*66/100)
trainData = data[:trainPortion]
testData = data[trainPortion:]

test_dataset = testData[:, :-3].tolist()
test_target = testData[:, -3:].tolist()

dataset = trainData[:, :-3].tolist()
target = trainData[:, -3:].tolist()


myNet5 = net.Dense()
myNet5.AddLayer(4, isInput=True)
myNet5.AddLayer(5, activationFunction=af.Sigmoid)
myNet5.AddLayer(3, activationFunction=af.Sigmoid)

myNet20 = net.Dense()
myNet20.AddLayer(4, isInput=True)
myNet20.AddLayer(20, activationFunction=af.Sigmoid)
myNet20.AddLayer(3, activationFunction=af.Sigmoid)

start_time = time.time()
myNet5.Train(dataset, target, iterationCount= 260, learningRateStart=.6, learningRateEnd=.0, regulationRate=.02)
train5_Time = time.time() - start_time
start_time = time.time()
myNet20.Train(dataset, target, iterationCount= 260, learningRateStart=.6, learningRateEnd=.0, regulationRate=.02)
train20_Time = time.time() - start_time

# **************************************** Print Metrics *************************************************************
t5, o5= ev.MultiClassToOne(myNet5, test_dataset, test_target)
t20, o20= ev.MultiClassToOne(myNet20, test_dataset, test_target)
print(" ")
print("---------------------------------------------------------------------------------")
print("---------------------------------- IRIS Data Set --------------------------------")

ev.PrintMetrics("5 hidden units",t5,o5,train5_Time)
print("-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -")
ev.PrintMetrics("20 hidden units",t20,o20,train20_Time)

print("---------------------------------------------------------------------------------")
print(" ")

