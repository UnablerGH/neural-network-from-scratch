import neural_network as nn

input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.3

# Creating basic neural network

n = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
print(n)