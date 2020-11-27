#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main module to train the network for the game
"""
from multiprocessing import Pool

from numpy import argmin, array, load
from pandas import read_csv
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork

split_dataset = lambda dataset: [dataset[:, [0, 1]], dataset[:, [2, 3]]]
get_arch_file = lambda layer: f'model_weights_{layer}_h_layer'
get_arch_numpy = lambda layer: f'weights/{get_arch_file(layer)}.npy'
get_arch_png = lambda layer: f'errors/{get_arch_file(layer)}.png'

training_dataset = split_dataset(array(read_csv('./training_data.csv', header=None)))
validation_dataset = split_dataset(array(read_csv('./validation_data.csv', header=None)))

def training(net):
    """
    Trains the neural network

    @type net: NeuralNetwork
    """

    rmses, val_rmses = [], []
    try:
        cur_rmses, cur_val_rmses = net.fit(training_dataset, validation_dataset, 20)
        rmses += cur_rmses[1:]
        val_rmses += cur_val_rmses[1:]

        # incomplete_trainings = 0
        # while True:
        #     epochs = 10
        #     print(f'Learning rate: {net.learning_rate}')
        #     cur_rmses, cur_val_rmses = net.fit(training_dataset, validation_dataset, epochs)
        #     rmses += cur_rmses[1:]
        #     val_rmses += cur_val_rmses[1:]
        #     if len(cur_rmses) < epochs + 1:
        #         net.learning_rate /= 2
        #         incomplete_trainings += 1
        #     else:
        #         net.learning_rate *= 1.1
        #         incomplete_trainings = 0

    except KeyboardInterrupt:
        print()

    print(f'Final RMSE: {rmses[-1]:.4%}, VAL RMSE: {val_rmses[-1]:.4%}')

    return rmses, val_rmses

def create_network_from_layers(layers):
    assert len(layers) >= 1, 'Layers array is empty :('

    network = NeuralNetwork()
    network.add_layer(layers[0], 2)
    for index in range(1, len(layers)):
        network.add_layer(layers[index])
    network.add_layer(2)
    return network

def create_network_permutations(layer_count, networks=None, current_layers=None):
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

def train_layer(network):
    print(f'Training:\n{network}')
    rmses, val_rmses = training(network)
    return (rmses[-1] + val_rmses[-1]) / 2

def train_layer_networks(networks, pool):
    return array(pool.map(train_layer, networks))

def decide_layer_config(layer_count, pool):
    networks = create_network_permutations(layer_count)
    final_rmses = train_layer_networks(networks, pool)
    best_network_index = argmin(final_rmses)
    best_network = networks[best_network_index]
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

def create_network_from_filename(filename):
    """
    Create network based on the weights stored in filename

    @type filename: str
    """
    weights = load(filename, allow_pickle=True)
    return NeuralNetwork(weights, 2)

def save_best_layer(net):
    """
    Saves the best neural network of this architecture based on the layer

    @type net: NeuralNetwork
    """
    architecture_filename = get_arch_numpy(len(net.layers) - 1)
    print(f'Saving {architecture_filename}')
    try:
        current_best_network = create_network_from_filename(architecture_filename)
        net_rmse = net.calculate_rmse(training_dataset)
        best_rmse = current_best_network.calculate_rmse(training_dataset)
        if net_rmse < best_rmse:
            print('New network is better than previous')
            net.save(architecture_filename)
        else:
            print('Old network was better...')
    except FileNotFoundError:
        print('New network is a new architecture')
        net.save(architecture_filename)

def decide_best_from_array(networks):
    """
    Based on a list of networks decide which is the best,
    being classified as the best the one with the lowest
    rmse
    """
    best_network = networks[0]
    best_rmse = best_network.calculate_rmse(training_dataset)
    print(f'{best_network}\nRMSE {best_rmse:.4%}\n')
    for index in range(1, len(networks)):
        current_network = networks[index]
        current_rmse = current_network.calculate_rmse(training_dataset)
        print(f'{current_network}\nRMSE {current_rmse:.4%}\n')
        if best_rmse > current_rmse:
            best_network = current_network
            best_rmse = current_rmse
    return best_network, best_rmse

def decide_best_architecture():
    """
    Decides which is the best network based on having the lowest rmse
    """
    print('Deciding best architecture\n')
    max_current_layer = 4
    filenames = list(map(get_arch_numpy, list(range(1, max_current_layer + 1))))
    networks = list(map(create_network_from_filename, filenames))
    best_network, best_rmse = decide_best_from_array(networks)

    print(f'{best_network}\nRMSE {best_rmse:.4%}\n')
    best_network.save('model_weights.npy')

def main(pool):
    """Main procedure for building NN"""
    # net = create_network_from_filename(get_arch_numpy(4))

    best_networks = []
    for index in range(3, 5):
        # net = create_network_from_filename(get_arch_numpy(index))
        net = decide_layer_config(index, pool)
        save_best_layer(net)
        best_networks.append(net)

    for net in best_networks:
        print(net)

    # print('Current architecture\n', net)

    # rmses, val_rmses = training(net)
    # plot(rmses, val_rmses, len(net.layers) - 1)

    # save_best_layer(net)

    decide_best_architecture()

if __name__ == "__main__":
    main(Pool())
