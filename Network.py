import random as rand


class Node:
    def __init__(self, activationFunction, isInput=False):
        self.activationFunction = activationFunction
        self.inEdges = []
        self.outEdges = []
        self.output = -1
        self.isInput = isInput
        self.delta = 0  # Error of this node

    def Output(self, reCompute=False):
        if reCompute and (not self.isInput):
            self.output = self.activationFunction.Value(self, reCompute)

        return self.output

    def GetDelta(self, targetValue, isOutput=False):
        error = 0
        if isOutput:
            # For each network output unit k, calculate its error term
            error = targetValue - self.Output(False)
        else:
            # For each hidden unit h, calculate its error term
            error = sum((element.weight * element.outNode.delta)
                        for element in self.outEdges)  # sum of weighted delta of next layer

        self.delta = self.activationFunction.Derivative(self)*error
        return self.delta


class Edge:
    def __init__(self, inNode, outNode):
        self.inNode = inNode
        self.outNode = outNode
        self.weight = rand.uniform(-1.0, 1.0)

    def UpdateWeight(self, learningRate, targetValue=None, isOutputLayer=False):
        # learning rate *             Delta of nex layer's node                     *   xji
        delta_W = learningRate * self.outNode.GetDelta(
            targetValue, isOutput=isOutputLayer) * self.inNode.Output(reCompute=False)
        self.weight = self.weight + delta_W


class Dense:
    def __init__(self):
        self.layers = []

    def AddLayer(self, nodeCount, activationFunction=None, isInput=False):
        newLayer = []
        if isInput:
            for i in range(nodeCount):
                newLayer.append(Node(activationFunction, isInput=True))
            self.layers.append(newLayer)
        else:
            if len(self.layers) == 0:
                raise Exception('First Layer Sould be Input')
            else:
                for j in range(nodeCount):
                    nodeJ = Node(activationFunction)
                    for i in range(len(self.layers[-1])):
                        nodeI = self.layers[-1][i]
                        edgeJI = Edge(nodeI, nodeJ)

                        nodeI.outEdges.append(edgeJI)
                        nodeJ.inEdges.append(edgeJI)

                    newLayer.append(nodeJ)

                self.layers.append(newLayer)

    def GetOutput(self, inputData):
        inputLayer = self.layers[0]
        if len(inputData) != len(inputLayer):
            raise Exception('Input Data is in different size with input layer')
        else:
            for i in range(len(inputLayer)):
                inputLayer[i].output = inputData[i]

            return [node.Output(reCompute=True) for node in self.layers[-1]]

    def BackPropagate(self, input, target, learningRate):
        if len(input) != len(self.layers[0]):
            raise Exception('Input Data is in different size with input layer')
            return
        if len(target) != len(self.layers[-1]):
            raise Exception('Output layer is in different size with target')
            return

        # Forward Propagate
        self.GetOutput(input)

        # Back Propagate
        for l in range(-1, -len(self.layers), -1):
            for j in range(len(self.layers[l])):
                nodeJ = self.layers[l][j]
                for i in range(len(self.layers[l-1])):
                    W_ji = nodeJ.inEdges[i]
                    if (l == -1):  # if this is the output layer
                        W_ji.UpdateWeight(
                            learningRate, targetValue=target[j], isOutputLayer=True)
                    else:
                        W_ji.UpdateWeight(learningRate)

    def Train(self, dataset, targets, iterationCount, learningRateStart, learningRateEnd, regulationRate=0):
        learningRate = learningRateStart
        decayStep = (learningRateStart - learningRateEnd)/iterationCount
        for i in range(iterationCount):
            for j in range(len(dataset)):
                self.BackPropagate(dataset[j], targets[j], learningRate)
            learningRate -= decayStep

            if regulationRate > 0:
                self.MultiplyAllWeights((1 - (2*learningRate*regulationRate)))

            print("Iteration Count: {} ".format(i), end="\r", flush=True)
        print(" ")

    def MultiplyAllWeights(self, coefficient):
        for l in range(-1, -len(self.layers), -1):
            for j in range(len(self.layers[l])):
                nodeJ = self.layers[l][j]
                for i in range(len(self.layers[l-1])):
                    W_ji = nodeJ.inEdges[i]
                    W_ji.weight *= coefficient
