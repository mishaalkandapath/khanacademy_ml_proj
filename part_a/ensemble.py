from utils_a import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

# import matplotlib.pyplot as plt
import part_a.neural_network as nn
import knn
import item_response

# cursor parking spot:

# sample users

def load_data(base_path="data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data

def evaluate_ensemble(valid_data, zero_train_data, train_data, nn_model, knn_model, irt_models):
    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(zero_train_data[u]).unsqueeze(0)
        nn_output = nn_model(inputs)

        nn_guess = nn_output[0][valid_data["question_id"][i]].item()
        knn_guess = knn_model[u][valid_data["question_id"][i]]
        irt_guess = item_response.sigmoid((irt_models[0][u] - irt_models[1][valid_data["question_id"][i]]).sum())

        guess = (nn_guess + knn_guess + irt_guess) / 3

        #embed it back intpo the matrix
        zero_train_data[u][valid_data["question_id"][i]] = guess

    #evaluate the matrix
    return sparse_matrix_evaluate(zero_train_data, valid_data)

def main():
    zero_train, train, valid, test = load_data()
    
    # train generate predictions
    # train on the neural network:
    nn_model = nn.AutoEncoder(train.shape[1], 100)
    nn.train(nn_model, 0.1, 0.01, train, zero_train, valid, 100)
        
    knn_matrix = knn.knn_impute_ensemble(train, 10)
    
    irt_output = item_response.irt_ensemble(train, .2, 50)
    
    # # average and evaluate
    valid_acc = evaluate_ensemble(valid, zero_train, train, nn_model, knn_matrix, irt_output)
    test_acc = evaluate_ensemble(test, zero_train, train, nn_model, knn_matrix, irt_output)

    print("Ensembled validation accuracy: {}".format(valid_acc))
    print("Ensembled test accuracy: {}".format(test_acc))

    
    
