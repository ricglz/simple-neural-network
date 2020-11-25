#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module containing the logic for layers"""
from numpy import array

try:
    from neuron import Neuron
except ImportError:
    from neural_network.neuron import Neuron

class Layer:
    """NN layer containing neurons"""
    delta = []
    error = []
    output = []

    def __init__(self, neurons_count, inputs_count, neuron_weights=None):
        if neuron_weights is not None and len(neuron_weights) != neurons_count:
            raise ValueError('Layer: Error with neuron_weights', neuron_weights, neurons_count)
        self.inputs_count = inputs_count
        create_neuron = lambda index: Neuron(inputs_count)
        if neuron_weights is not None:
            create_neuron = lambda index: Neuron(inputs_count, neuron_weights[index])
        self.neurons = list(map(create_neuron, range(neurons_count)))

    def __str__(self):
        return f'Inputs: {self.inputs_count}, Neurons: {len(self.neurons)}'

    def __map_neurons__(self, func):
        return array(list(map(func, self.neurons)))

    def calculate_outputs(self, inputs):
        """
        Return the outputs of each of the neurons of the layers
        based on inputs

        @return Numpy array containing the output of each of the neurons
        """
        calculate_output = lambda neuron: neuron.calculate_output(inputs)
        self.output = self.__map_neurons__(calculate_output)
        return self.output

    def get_weights(self):
        """Gets the weights of the neurons as a numpy array"""
        get_weights = lambda neuron: neuron.weights
        return self.__map_neurons__(get_weights)

    def act_func_derivs(self):
        get_derivs = lambda neuron: neuron.deriv_sigmoid()
        return self.__map_neurons__(get_derivs)

    def update_weights(self, inputs, learning_rate=0.13):
        for index, neuron in enumerate(self.neurons):
            neuron.weights += self.delta[index] * inputs * learning_rate

if __name__ == "__main__":
    layer = Layer(3, 2)
    weights = layer.get_weights()
    print(weights, len(weights) == 3)
    random_inputs = array([1, 2])
    outputs = layer.calculate_outputs(random_inputs)
    print(outputs, len(outputs) == 3)
