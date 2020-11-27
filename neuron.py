#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module containing the logic for neurons"""
from math import exp

from numpy import ndarray
from numpy.random import rand

def sigmoid(value):
    """Activation function, in this case the sigmoid function"""
    return 1 / (1 + exp(-value)) if value >= 0 else \
            1 - 1/(1 + exp(value))

class Neuron:
    """
    Overall class managing regarding a Neuron in Neural Network
    with sigmoid as the activation function
    """
    output = 0.0

    def __init__(self, weights_count, weights=None):
        self.weights = rand(weights_count) if weights is None else weights

    def calculate_output(self, inputs):
        """
        Calculate the output of the neuron based on the inputs
        and the weights and that each input has.
        @arg inputs: Is a numpy array containing the inputs received

        @return. Float value representing the value of that node
                 after perfoming an activation function and summing the
                 input times the value
        """
        if not isinstance(inputs, (list, ndarray)):
            raise ValueError('Neuron: Inputs are not lists', inputs)
        if not isinstance(self.weights, (list, ndarray)):
            raise ValueError('Neuron: Weights are not lists', self.weights)
        if len(inputs) != len(self.weights):
            raise ValueError('Neuron: Not correct length of inputs', inputs, self.weights)
        weighted_inputs = self.weights * inputs
        weighted_inputs_sum = weighted_inputs.sum()
        self.output = sigmoid(weighted_inputs_sum)
        return self.output

    def deriv_sigmoid(self):
        """Derivative value of the sigmoid function based on the value"""
        return self.output * (1 - self.output)

if __name__ == "__main__":
    neuron = Neuron(3)
    random_inputs = rand(3)
    print('Random inputs', random_inputs)
    output = neuron.calculate_output(random_inputs)
    print('Output', output)
