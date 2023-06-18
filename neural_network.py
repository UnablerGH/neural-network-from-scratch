# neural network class definition
import numpy as np
# scipy.special for the sigmoid function expit()
import scipy.special

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
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train the neural network

    def train(self, inputs_list, targets_list):
        # convert inputs list to 2D array
        inputs = np.array(inputs_list, ndmin=2).T
        # ndmin = 2 means that the array will be at least 2D, even if the input is 1D
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # apply sigmoid function to hidden outputs
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # apply sigmoid function to final output
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        np.transpose(inputs))
        
        pass


    
    # query the neural network
    def query(self, inputs_list):
        inputs = inputs = np.array(inputs_list, ndmin=2).T
        # ndmin = 2 means that the array will be at least 2D, even if the input is 1D

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # apply sigmoid function to hidden outputs
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # apply sigmoid function to final output
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
