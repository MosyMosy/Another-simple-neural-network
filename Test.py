import Network as net
import ActivationFunction as af

myNet = net.Dense()
myNet.AddLayer(2,isInput=True)
myNet.AddLayer(3,activationFunction= af.Sigmoid)
myNet.AddLayer(1,activationFunction= af.Sigmoid)

dataset = [[5,1],[8,2],[9,.5],[7,1.2],[.5,8],[1.2,9.5],[.7,7],[1.5,6]]
target  = [[0],[0],[0],[0],[1],[1],[1],[1]]

myNet.Train(dataset,target,1000,.2)

print(myNet.GetOutput([6.5,1.5]))
print(myNet.GetOutput([.1,10]))
