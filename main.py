#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main module to train the network for the game
"""
from multiprocessing import Pool

from numpy import argmin, array
from pandas import read_csv
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork

split_dataset = lambda dataset: [dataset[:, [0, 1]], dataset[:, [2, 3]]]
get_arch_file = lambda layer: f'model_weights_{layer}_h_layer'
get_arch_numpy = lambda layer: f'weights/{get_arch_file(layer)}.npy'
get_arch_png = lambda layer: f'errors/{get_arch_file(layer)}.png'
average_error = lambda a, b: (a + b) / 2

training_dataset = split_dataset(array(read_csv('./training_data.csv', header=None)))
validation_dataset = split_dataset(array(read_csv('./validation_data.csv', header=None)))
pool = None

def training(net):
    """
    Trains a neural network and return the rmse of training
    and validation

    @type net: NeuralNetwork
    """

    rmses, val_rmses = [], []
    try:
        # cur_rmses, cur_val_rmses = net.fit(training_dataset, validation_dataset, 20)
        # rmses += cur_rmses[1:]
        # val_rmses += cur_val_rmses[1:]

        incomplete_trainings = 0
        while incomplete_trainings < 6:
            epochs = 15
            print(f'Learning rate: {net.learning_rate}')
            cur_rmses, cur_val_rmses = net.fit(training_dataset, validation_dataset, epochs)
            rmses += cur_rmses[1:]
            val_rmses += cur_val_rmses[1:]
            if len(cur_rmses) < epochs + 1:
                net.learning_rate /= 2
                incomplete_trainings += 1
            else:
                net.learning_rate *= 1.1
                incomplete_trainings = 0

    except KeyboardInterrupt:
        print()

    print(f'Final RMSE: {rmses[-1]:.4%}, VAL RMSE: {val_rmses[-1]:.4%}')

    return rmses, val_rmses

def create_network_from_layers(layers):
    """Given an array of the neurons per layer, create a neural network"""
    assert len(layers) >= 1, 'Layers array is empty :('

    network = NeuralNetwork()
    network.add_layer(layers[0], 2)
    for index in range(1, len(layers)):
        network.add_layer(layers[index])
    network.add_layer(2)
    return network

def create_network_permutations(layer_count, networks=None, current_layers=None):
    """Recursive function to create all the permutations of networks possible"""
    if current_layers is None:
        current_layers = []
        networks = []
    for value in range(5, 11):
        new_layers = current_layers + [value]
        if layer_count == 1:
            # print('Creating new network: ', new_layers)
            network = create_network_from_layers(new_layers)
            networks.append(network)
        else:
            create_network_permutations(layer_count - 1, networks, new_layers)
    return networks

def train_and_get_error(network):
    """
    @type network: NeuralNetwork
    """
    rmses, val_rmses = training(network)
    return average_error(rmses[-1], val_rmses[-1])

def neural_network_avg_error(network):
    """Returns the average error of the neural network"""
    rmse = network.calculate_rmse(training_dataset)
    val_rmse = network.calculate_rmse(validation_dataset)
    return average_error(rmse, val_rmse)

def train_or_get_errors(networks, train):
    """Train or get errors of the networks in a concurrent way"""
    map_func = train_and_get_error if train else neural_network_avg_error
    return array([map_func(network) for network in networks])
    # return array(pool.map(map_func, networks))

def decide_best_from_array(networks, train=False):
    """
    Based on a list of networks decide which is the best,
    being classified as the best the one with the lowest
    rmse
    """
    average_rmses = train_or_get_errors(networks, train)
    index = argmin(average_rmses)
    return networks[index], average_rmses[index]

def decide_layer_config(layer_count):
    """
    Decide the best permutation of the network based on the
    amount of hidden layers
    """
    networks = create_network_permutations(layer_count)
    best_network, _ = decide_best_from_array(networks)
    print(f'Best {layer_count} layers network\n{best_network}')
    return best_network

def plot(rmses, val_rmses, h_layers):
    """
    Allows to plot the rmse of the training and validation dataset

    @type rmses: list
    @type val_rmses: list
    @type h_layers: number
    """
    plt.plot(rmses, label='Training dataset RMSE')
    plt.plot(val_rmses, label='Validation dataset RMSE')

    plt.legend(loc='upper left')

    image_name = get_arch_png(h_layers)
    plt.savefig(image_name)

    plt.show()

def save_best_layer(net):
    """
    Saves the best neural network of this architecture based on the layer

    @type net: NeuralNetwork
    """
    architecture_filename = get_arch_numpy(len(net.layers) - 1)
    print(f'Saving {architecture_filename}')
    try:
        current_best_network = NeuralNetwork.game_neural_network(architecture_filename)
        net_rmse = net.calculate_rmse(training_dataset)
        best_rmse = current_best_network.calculate_rmse(training_dataset)
        print(f'Current best error: {best_rmse:.4%}, Network error: {net_rmse:.4%}')
        if net_rmse < best_rmse:
            print('New network is better than previous')
            net.save(architecture_filename)
        else:
            print('Old network was better...')
    except FileNotFoundError:
        print('New network is a new architecture')
        net.save(architecture_filename)

def decide_best_architecture():
    """
    Decides which is the best network based on having the lowest rmse
    """
    print('Deciding best architecture\n')
    max_current_layer = 4
    filenames = list(map(get_arch_numpy, list(range(1, max_current_layer + 1))))
    networks = list(map(NeuralNetwork.game_neural_network, filenames))
    best_network, best_rmse = decide_best_from_array(networks, train=False)

    print(f'{best_network}\nRMSE {best_rmse:.4%}\n')
    best_network.save('model_weights.npy')

def main():
    """Main procedure for building NN"""
    # net = NeuralNetwork.game_neural_network(get_arch_numpy(4))

    # networks = [NeuralNetwork.game_neural_network(get_arch_numpy(index)) for index in range(1, 5)]

    # best_layer_architecture = {
    #     1: [10],
    #     2: [9, 6],
    #     3: [6, 8, 10],
    #     4: [10, 5, 10, 5]
    # }
    # layer_neurons = [best_layer_architecture[index] for index in range(1, 5)]
    # networks = list(map(create_network_from_layers, layer_neurons))

    # train_or_get_errors(networks, train=True)
    # for network in networks:
    #     save_best_layer(network)

    # print('Current architecture\n', net)

    # rmses, val_rmses = training(net)
    # plot(rmses, val_rmses, len(net.layers) - 1)

    # save_best_layer(net)

    decide_best_architecture()

if __name__ == "__main__":
    pool = Pool()
    main()
