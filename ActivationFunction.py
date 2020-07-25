import math

class AFInterface:
    @staticmethod
    def Value(node,reCompute):
        pass

    @staticmethod
    def Derivative(node):
        pass

class Linear(AFInterface):
    @staticmethod
    def Value(node,reCompute):
        # sum of products
        return sum((element.weight*element.inNode.Output(reCompute)) for element in  node.inEdges)

    @staticmethod
    def Derivative(node):
        return 1


class Sigmoid(AFInterface):
    @staticmethod
    def Value(node,reCompute):
        return 1/(1+ math.exp(-Linear.Value(node,reCompute)))

    @staticmethod
    def Derivative(node):
        _value = Sigmoid.Value(node,False)
        return _value*(1-_value)



class ReLU(AFInterface):
    @staticmethod
    def Value(node,reCompute):
        x = Linear.Value(node,reCompute)
        if x < 0:
            return 0
        else:
            return x

    @staticmethod
    def Derivative(node):
        x = Linear.Value(node,False)
        if x < 0:
            return 0
        else:
            return 1