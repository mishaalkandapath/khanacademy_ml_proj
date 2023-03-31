from utils_a import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt

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

class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, k+50)
        self.f = nn.Linear(k+50, num_question)
        self.dropout = nn.Dropout(p=0.5)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs
        out = F.sigmoid(self.g(out))
        # out = self.dropout(out)
        out = F.sigmoid(self.h(out))
        out = self.dropout(out)
        out = self.f(out)
        out = F.sigmoid(out)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, k):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 

    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr) #optim.Adam(model.parameters(), lr=lr, weight_decay=0.001, amsgrad=True)
                            #
    num_student = train_data.shape[0]

    #storing data for plotting
    train_loss_arr = []
    valid_acc_arr = []

    for epoch in range(0, num_epoch):
        # optimizer.param_groups[0]['lr'] = lr * (0.1 ** (epoch // 10))
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + lamb * model.get_weight_norm() #added the weight regularizer,. 
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        train_loss_arr.append(train_loss)
        valid_acc_arr.append(valid_acc)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    plot_loss(train_loss_arr, valid_acc_arr, k)

    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)

def plot_loss(train_loss, valid_acc, k):
    """ Plot the loss and accuracy curves.

    :param train_loss: list
    :param valid_acc: list
    :return: None
    """
    fig, ax1 = plt.subplots()
    ax1.plot(train_loss, 'b-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(valid_acc, 'r-')
    ax2.set_ylabel('Validation Accuracy', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.savefig("loss_{}_modified.png".format(k))


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k = 100
    

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 90
    lamb = 0.001

    for k in [10, 50, 100, 200, 500]:
        model = AutoEncoder(zero_train_matrix.shape[1], k)
        print()
        print("k = {}, lamb = {}".format(k, lamb))
        train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, k)
        test_acc = evaluate(model, zero_train_matrix, test_data)
        

    # choosing k = 100 based on validation accuracy of 0.657, testacc 0.6627
    # for lamb in [0.001, 0.01, 0.1, 1]:
    #     model = AutoEncoder(zero_train_matrix.shape[1], k)
    #     train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, k)
    #     test_acc = evaluate(model, zero_train_matrix, test_data)
    #     print("Test Accuracy: {}".format(test_acc))
    # the model performs significantly worse in terms of loss, less but still worse in terms of validation accuracy
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

"""
Difference betqween ALS and Neural networks:
ALS is a matrix factorization method. Neural networks are machine learning algorithms
ALS is used best for best for filling in missing values in a matrix, whereas neural networks are used for a large variety of tasks.
als is linear, whereas neural nets are non-linear
ALS is more interprertable
more computationally efficient

"""


if __name__ == "__main__":
    main()


"""
Observations:
using adam more stable ppredictions, stabilized in range of 68.3 fdoe k =10, 0.001
using dropout with p=0.5 gave sufficient stability in 0.69 range. 
using dropout with adam was distrous, gave 0.618
sgd with momentum did not work

added an extra layer in nn, did dropout on both layers and thot got us into the 70s
afterwards, set lr to 0.01 and then gradually decrease learning rate and that stabilized us in the 70s
afterwards dropout was removed from the outer layer and we got a small 4th decimal boost in accuracy
epcoch 74, 0.723 was the best we ever got at this stage, by dropping out only the outer layer, and not the inner layer
"""



"""
Final modified model stats:
best K = 10
valid acc = 0.7020886254586508
test_acc = 0.7019475021168501
"""