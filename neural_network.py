#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module containing the logic for the overall neural network"""
from math import sqrt
from numpy import array, load, matmul, mean, ndarray, save, square

try:
    from layer import Layer
except ImportError:
    from neural_network.layer import Layer

class NeuralNetwork:
    def __init__(self, layer_weights=None, inputs_count=None):
        self.layers = []
        self.previous_size = None
        self.learning_rate = 5
        if layer_weights is not None:
            self.__initialize_layers__(layer_weights, inputs_count)

    def __str__(self):
        layer_description = lambda enum: f'Layer {enum[0]} - {enum[1]}'
        layer_descriptions = list(map(layer_description, enumerate(self.layers)))
        return '\n'.join(layer_descriptions)

    @staticmethod
    def game_neural_network(filepath='model_weights.npy'):
        """Create the neural network based on the data and inputs for the project"""
        loaded_weights = load(filepath, allow_pickle=True)
        network = NeuralNetwork(loaded_weights, inputs_count=2)
        return network

    def add_layer(self, neurons_count, inputs_count=None, layer_weights=None):
        """Add a new layer to the NeuralNetwork"""
        current_size = self.previous_size if self.previous_size is not None else inputs_count
        if current_size is None:
            raise ValueError('NeuralNetwork: current_size is None')
        layer = Layer(neurons_count, current_size, layer_weights)
        self.previous_size = neurons_count
        self.layers.append(layer)

    def __initialize_layers__(self, layer_weights, inputs_count):
        self.previous_size = inputs_count
        for layer_weight in layer_weights:
            self.add_layer(len(layer_weight), layer_weights=layer_weight)

    def __feed_forward__(self, inputs):
        """
        Given an array of inputs, calculate the outputs based on the
        current layers.
        @return numpy array containing the output of the last layer
        """
        current_inputs = array(inputs)
        for layer in self.layers:
            current_inputs = layer.calculate_outputs(current_inputs)
        return current_inputs

    def feed_forward(self, inputs):
        """
        Given an array of inputs, calculate the outputs based on the
        current layers.
        @return numpy array containing the output of the last layer
        """
        type_of_array = isinstance(inputs[0], (list, ndarray))
        return array(list(map(self.__feed_forward__, inputs))) if type_of_array \
                else self.__feed_forward__(inputs)

    def get_weights(self):
        """Get weights of all the neurons of all the layers in the NN"""
        get_weights = lambda layer: layer.get_weights()
        return array(list(map(get_weights, self.layers)), dtype=object)

    def calculate_rmse(self, dataset):
        """Calculate the mean square error of a dataset"""
        inputs, real_output = dataset
        output = self.feed_forward(inputs)
        rmse = sqrt(mean(square(real_output - output)))
        return rmse

    def __calculate_errors__(self, inputs, real_output):
        reversed_layers = list(reversed(self.layers))
        for index, layer in enumerate(reversed_layers):
            if index == 0:
                layer.error = real_output - self.feed_forward(inputs)
            else:
                prev_layer = reversed_layers[index - 1]
                layer.error = matmul(prev_layer.get_weights().T, prev_layer.delta)
            layer.delta = layer.error * layer.act_func_derivs()

    def backpropagation(self, inputs, real_output):
        """Performs a backpropagation"""
        self.__calculate_errors__(inputs, real_output)
        for index, layer in enumerate(self.layers):
            actual_inputs = inputs if index == 0 else self.layers[index - 1].output
            layer.update_weights(actual_inputs, self.learning_rate)

    def __run_epoch__(self, training_data, validation_data, rmses, val_rmses):
        training_x, training_y = training_data
        for index, inputs in enumerate(training_x):
            self.backpropagation(inputs, training_y[index])
        rmse = self.calculate_rmse(training_data)
        rmses.append(rmse)
        val_rmse = self.calculate_rmse(validation_data)
        val_rmses.append(val_rmse)

    def fit(self, training_data, validation_data, epochs=20):
        """
        @type training_data: (ndarray, ndarray)
        @type validation_data: (ndarray, ndarray)
        @type epochs: int
        @rtype (list, list)
        """
        epoch = 1
        rmses = [self.calculate_rmse(training_data)]
        val_rmses = [self.calculate_rmse(validation_data)]
        print(f'Epoch 0, rmse: {rmses[-1]:.4%}, val rmse: {val_rmses[-1]:.4%}')
        decreasing_error, significant_error = True, True
        while epoch <= epochs and decreasing_error and significant_error:
            self.__run_epoch__(training_data, validation_data, rmses, val_rmses)
            print(f'Epoch {epoch}, rmse: {rmses[-1]:.4%}, val rmse: {val_rmses[-1]:.4%}')
            epoch +=1
            decreasing_error = rmses[-2] > rmses[-1] and val_rmses[-2] > val_rmses[-1]
            significant_error = (rmses[-2] - rmses[-1]) >= 1e-6
        return rmses, val_rmses

    def save(self, name):
        """
        Save the final weights of the NN in an specific file
        @type name: string
        """
        final_weights = self.get_weights()
        save(name, final_weights)
        return final_weights

if __name__ == "__main__":
    net = NeuralNetwork()
    net.add_layer(3, 2)
    net.add_layer(3)
    net.add_layer(2)

    x = array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = array([[0], [0], [0], [1]])

    net.fit(x, y)

    print(net.save('test_weights'))

    weights = load('test_weights.npy', allow_pickle=True)
    net = NeuralNetwork(weights, 2)
    print(weights)

    print(net.feed_forward(x))
