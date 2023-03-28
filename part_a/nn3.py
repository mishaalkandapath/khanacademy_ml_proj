from utils_a import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt
from scipy.sparse import load_npz, save_npz, csr_matrix

subjects = []

def load_subjects():
    """Load all subject ids""" 
    global subjects
    with open("data/subject_meta.csv") as f:
        for line in f:
            line = line.strip().split(",")
            if line[0] != "subject_id":
                subjects.append(int(line[0]))
    subjects = sorted(list(set(subjects)))
                
def question_subject_metadata():
    """ given a question id, return all the subject ids in the question

    :return: A dictionary {question_id: subject_id}
    """
    global subjects
    topic_areas = {}
    with open("data/question_meta.csv") as f:
        for line in f:
            line = line.strip().split(",")
            if line[0] != "question_id":
                question_id = int(line[0])
                line = line[1:]
                topics = []
                for idx, element in enumerate(line):
                    #extract only the numbers out of all the characters and add to topics
                    number = ""
                    for char in element:
                        if char.isdigit():
                            number += char
                    topics.append(int(number))
                topic_areas[question_id] = topics
    return topic_areas

def load_student_metaadata():
    """ Return a numpy matrix of student metadata in order of student ids"""
    global subjects
    student_metadata = np.zeros((542, 2))
    with open("data/student_meta.csv") as f:
        for line in f:
            line = line.strip().split(",")
            if line[0] != "user_id":
                user_id = int(line[0])
                gender = int(line[1])
                student_metadata[user_id][0] = user_id
                student_metadata[user_id][1] = gender
    return student_metadata

def assemble_new_data(train_data):
    """ Assemble the new data matrix with student metadata and question subject metadata

    :return: A numpy matrix of the new data
    """
    global subjects
    student_metadata = load_student_metaadata()
    question_subjects = question_subject_metadata()
    new_data = np.zeros((542, len(subjects)+2))
    new_data = -1 * new_data
    for i in range(542):
        new_data[i][0] = student_metadata[i][0]
        new_data[i][1] = student_metadata[i][1]
        question_correctness_data = {}
        for j in range(len(question_subjects)):
            for k in range(len(question_subjects[j])):
                if question_subjects[j][k] not in question_correctness_data:
                    question_correctness_data[question_subjects[j][k]] = [0,0]
                if train_data[i][j] == 1:
                    question_correctness_data[question_subjects[j][k]][0] += 1
                elif train_data[i][j] == 0:
                    question_correctness_data[question_subjects[j][k]][1] += 1
        for j in range(len(subjects)):
            if subjects[j] in question_correctness_data:
                if question_correctness_data[subjects[j]][0] == 0:
                    new_data[i][j+2] = -1
                elif question_correctness_data[subjects[j]][1] == 0:
                    new_data[i][j+2] = 1
                else:
                    new_data[i][j+2] = question_correctness_data[subjects[j]][0] / (question_correctness_data[subjects[j]][1])
    #store the matrix as npz
    new_data = csr_matrix(new_data)
    save_npz("data/new_data.npz", new_data)
    return new_data 

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


class Autoenc(torch.nn.Module):
   def __init__(self):
      super().__init__()

      self.encoder = torch.nn.Sequential(
         torch.nn.Linear(1775, 1000),
         torch.nn.ReLU(),
         torch.nn.Linear(1000, 500),
         torch.nn.ReLU(),
         torch.nn.Linear(500, 100),
         torch.nn.ReLU(),
         torch.nn.Linear(100, 50)
      )

      self.decoder = torch.nn.Sequential(
         torch.nn.Linear(50, 100),
         torch.nn.ReLU(),
         torch.nn.Linear(100, 500),
         torch.nn.ReLU(),
         torch.nn.Linear(500, 1000),
         torch.nn.ReLU(),
         torch.nn.Linear(1000, 1775),
         torch.nn.Sigmoid()
      )
   def forward(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
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
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    #storing data for plotting
    train_loss_arr = []
    valid_acc_arr = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) #lamb * model.get_weight_norm() #added the weight regularizer,. 
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        train_loss_arr.append(train_loss)
        valid_acc_arr.append(valid_acc)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    # plot_loss(train_loss_arr, valid_acc_arr)

    
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

def plot_loss(train_loss, valid_acc):
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
    plt.savefig("loss.png")


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #load the sparse augmented data:
    # load_subjects()
    path = os.path.join("data", "new_data.npz")
    # aug_matrix = assemble_new_data(train_matrix)
    aug_matrix = load_npz(path)
    aug_matrix = aug_matrix.toarray()
    aug_matrix = aug_matrix[:, 2:3]
    aug_matrix = np.concatenate((train_matrix, aug_matrix), axis=1)
    zero_train_matrix = aug_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(aug_matrix)] = 0
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    aug_matrix = torch.FloatTensor(aug_matrix)

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k = 100
    

    # Set optimization hyperparameters.
    lr = 0.1
    num_epoch = 100
    lamb = 0

    # for k in [10, 50, 100, 200, 500]:
    #     model = AutoEncoder(train_matrix.shape[1], k)
    #     print()
    #     print("k = {}".format(k))
    #     train(model, lr, lamb, train_matrix, zero_train_matrix,
    #       valid_data, num_epoch)

    #choosing k = 100 based on validation accuracy of 0.657, testacc 0.6627
    for lamb in [0.001, 0.01, 0.1, 1]:
        model = Autoenc()
        train(model, lr, lamb, aug_matrix, zero_train_matrix, valid_data, k)
        test_acc = evaluate(model, zero_train_matrix, test_data)
        print("Test Accuracy: {}".format(test_acc))
    #the model performs significantly worse in terms of loss, less but still worse in terms of validation accuracy
    
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