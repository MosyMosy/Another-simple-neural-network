import Network as net
import ActivationFunction as af
import Evaluation as ev
import numpy as np
import os
import math
import time
from sklearn.linear_model import LinearRegression


_DataSetPath = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'DataSet')   # Get the directory for StringFunctions
_curentdatapath = os.path.join(_DataSetPath, "winequality-red.data")


data = np.loadtxt(_curentdatapath, dtype=float, delimiter=",")
np.random.shuffle(data)

for i in range(len(data[0,:])):
    _max = np.max(data[:,i])
    _min = np.min(data[:,i])
    data[:,i] = (data[:,i]-_min)/(_max - _min)

trainPortion = int(len(data)*66/100)
trainData = data[:trainPortion]
testData = data[trainPortion:]

test_dataset = testData[:, :-1].tolist()
test_target = testData[:, -1:].tolist()

dataset = trainData[:, :-1].tolist()
target = trainData[:, -1:].tolist()


myNet5 = net.Dense()
myNet5.AddLayer(11, isInput=True)
# myNet5.AddLayer(5, activationFunction=af.Linear)
myNet5.AddLayer(1, activationFunction=af.Linear)

myNet20 = net.Dense()
myNet20.AddLayer(11, isInput=True)
myNet20.AddLayer(20, activationFunction=af.Linear)
myNet20.AddLayer(1, activationFunction=af.Sigmoid)

start_time = time.time()
myNet5.Train(dataset, target, iterationCount= 10, learningRateStart=.6, learningRateEnd=.0, regulationRate=.08)
train5_Time = time.time() - start_time
start_time = time.time()
myNet20.Train(dataset, target, iterationCount= 10, learningRateStart=.6, learningRateEnd=.0, regulationRate=.0)
train20_Time = time.time() - start_time

# **************************************** Print Metrics *************************************************************
from sklearn.metrics import mean_squared_error
t5, o5= ev.GetPredictions(myNet5, test_dataset, test_target)
t20, o20= ev.GetPredictions(myNet20, test_dataset, test_target)
print(" ")
print("---------------------------------------------------------------------------------")
print("--------------------------------- winequality-red -------------------------------")

print(">>>>>>>>>>>>> 5 hidden units")
print("mean_squared_error : {}".format(round(mean_squared_error(t5, o5),6))) 
print("Root_MSE           : {}".format(round(math.sqrt(mean_squared_error(t5, o5)),6)))     
print("Training Time      : {} Seconds".format(round(train5_Time,3)))
print("-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -")
print(">>>>>>>>>>>>> 20 hidden units")
print("mean_squared_error : {}".format(round(mean_squared_error(t20, o20),6)))
print("Root_MSE           : {}".format(round(math.sqrt(mean_squared_error(t20, o20)),6)))
print("Training Time      : {} Seconds".format(round(train20_Time,3)))

print("---------------------------------------------------------------------------------")
print(" ")



model = LinearRegression()
