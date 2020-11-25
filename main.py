#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main module to train the network for the game
"""
from numpy import array, load
from pandas import read_csv
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork

split_dataset = lambda dataset: [dataset[:, [0, 1]], dataset[:, [2, 3]]]
get_arch_file = lambda layer: f'model_weights_{layer}_h_layer'
get_arch_numpy = lambda layer: f'weights/{get_arch_file(layer)}.npy'
get_arch_png = lambda layer: f'errors/{get_arch_file(layer)}.png'

def training(net, training_dataset):
    """
    Trains the neural network

    @type net: NeuralNetwork
    @type training_dataset: list
    """
    validation_dataset = split_dataset(array(read_csv('./validation_data.csv', header=None)))

    rmses, val_rmses = [], []
    try:
        while True:
            epochs = 10
            print(f'Learning rate: {net.learning_rate}')

            cur_rmses, cur_val_rmses = net.fit(training_dataset, validation_dataset, epochs)
            rmses += cur_rmses[1:]
            val_rmses += cur_val_rmses[1:]

            if len(cur_rmses) < epochs + 1:
                net.learning_rate /= 2
            else:
                net.learning_rate *= 1.1

    except KeyboardInterrupt:
        print()

    print(f'Final RMSE: {net.calculate_rmse(training_dataset):.4%}')

    return rmses, val_rmses

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

def create_network(filename):
    """
    Create network based on the weights stored in filename
    @type filename: str
    """
    weights = load(filename, allow_pickle=True)
    return NeuralNetwork(weights, 2)

def save_best_layer(net, training_dataset):
    """
    Saves the best neural network of this architecture based on the layer

    @type net: NeuralNetwork
    """
    architecture_filename = get_arch_numpy(len(net.layers) - 1)
    print(f'Saving {architecture_filename}')
    try:
        current_best_network = create_network(architecture_filename)
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

def decide_best_architecture(training_dataset):
    """
    Decides which is the best network based on having the lowest rmse
    @type training_dataset: list
    """
    print('Deciding best architecture\n')
    max_current_layer = 4
    filenames = list(map(get_arch_numpy, list(range(1, max_current_layer + 1))))
    networks = list(map(create_network, filenames))

    best_network = networks[0]
    best_rmse = best_network.calculate_rmse(training_dataset)
    print(f'{best_network}\nRMSE {best_rmse:.4%}\n')
    for index in range(1, max_current_layer):
        current_network = networks[index]
        current_rmse = current_network.calculate_rmse(training_dataset)
        print(f'{current_network}\nRMSE {current_rmse:.4%}\n')
        if best_rmse > current_rmse:
            best_network = current_network
            best_rmse = current_rmse

    print(f'{best_network}\nRMSE {best_rmse:.4%}\n')
    best_network.save('model_weights.npy')

def main():
    """Main procedure for building NN"""
    # net = create_network(get_arch_numpy(4))
    training_dataset = split_dataset(array(read_csv('./training_data.csv', header=None)))

    for index in range(1, 5):
        net = create_network(get_arch_numpy(index))
        print(net)
        training(net, training_dataset)
        save_best_layer(net, training_dataset)

    # print('Current architecture\n', net)

    # rmses, val_rmses = training(net, training_dataset)
    # plot(rmses, val_rmses, len(net.layers) - 1)

    # save_best_layer(net, training_dataset)

    decide_best_architecture(training_dataset)

if __name__ == "__main__":
    main()
