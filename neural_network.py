# neural network class definition
import numpy as np

class neuralNetwork:
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr     = learningrate

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        """ We could do something like this:
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5), but we want to use a normal 
        distribution instead of a uniform distribution, because we want the weights to be centered around zero.
        We want the weights to be small, but not too small. We want them to be in the range of 1/sqrt(number of incoming links),
        so, we use pow(self.hnodes, -0.5) to get the square root of the number of incoming links to a hidden node.
        """    

    # train the neural network

    def train():
        pass
    
    # query the neural network
    def query():
        pass