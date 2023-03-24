from utils_a import *
from torch.autograd import Variable
from scipy.sparse import csr_matrix, save_npz

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt

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
                    new_data[i][j+2] = 0
                elif question_correctness_data[subjects[j]][1] == 0:
                    new_data[i][j+2] = 1
                else:
                    new_data[i][j+2] = question_correctness_data[subjects[j]][0] / (question_correctness_data[subjects[j]][0] + question_correctness_data[subjects[j]][1])
    # store the matrix as npz
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


class NN_model(nn.Module):
    def __init__(self, num_features, num_subjects, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(NN_model, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_features, k)
        self.h = nn.Linear(k, num_subjects)

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
        out = F.sigmoid(self.h(F.sigmoid(self.g(out))))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out
    
def train_b(model, lr, lamb, train_data, train_matrix, zero_train_data, valid_data, num_epoch):
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
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    
    #storing data for plotting
    train_loss_arr = []
    valid_acc_arr = []

    question_meta = question_subject_metadata()  # dictionary mapping questions to subject
    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(train_data[user_id]).unsqueeze(0)
            target = train_matrix[user_id]
            if inputs.shape[1] != 390:
                print("length wrong: " + str(inputs.shape[1]))
            # NOTE: all nan entries are to be ignored in loss calculation -- can't remove it bc indexing won't work properly
            # target[np.isnan(target)] = 0

            optimizer.zero_grad()
            output = model(inputs)
            
            output_questions = question_outcomes(output, target, question_meta)

            # Mask the target to only compute the gradient of valid entries.
            output_questions = np.where(np.isnan(target), 0, output_questions)
            target = np.where(np.isnan(target), 0, target)
            # nan_mask = np.isnan(train_matrix[user_id].unsqueeze(0).numpy())
            # target[0][nan_mask] = output_questions[0][nan_mask]
            
            # target[np.isnan(target)] = 0

            loss = torch.sum((torch.Tensor(output_questions) - torch.from_numpy(target)) ** 2.) + lamb * model.get_weight_norm()     # added the weight regularizer
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
        valid_acc = evaluate(model, train_data, valid_data, question_meta)
        train_loss_arr.append(train_loss)
        valid_acc_arr.append(valid_acc)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    plot_loss(train_loss_arr, valid_acc_arr)
    
def question_outcomes(output, target, question_meta):
    calculated_output = []  # length should match target when done
    # break all non-nan targets down into its subjects
    output = output.squeeze(0).detach().numpy()
    for i in range(len(target)):
        # find q in question_meta
        subjects = question_meta[i]
        # calculate predicted output: average of all subjects
        calculated_output.append(sum(output[x] for x in subjects)/len(subjects))
    # now mask the rest of this output with 0 where target was nan before comparing ?
    return np.asarray(calculated_output)  # maybe convert type before returning


def evaluate(model, train_data, valid_data, question_meta):
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
        question = question_meta[valid_data["question_id"][i]]
        output_np = output.squeeze(0).detach().numpy()
        guess = (sum(output_np[x] for x in question)/len(question)) >= 0.5
        # guess = output[0][valid_data["question_id"][i]].item() >= 0.5
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
    plt.savefig("part_b.png")

def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    load_subjects()
    global subjects
    train_sparse = load_train_sparse("data").toarray()
    # train_data = assemble_new_data(train_matrix)
    # theta = np.ones(5)*0.5
    # theta = csr_matrix(theta)
    # print(type(theta))
    # save_npz("data/theta.npz", theta)
    # print(load_new_sparse("data"))
    
    train_data = load_new_sparse("data").toarray()
    train_data = torch.from_numpy(load_new_sparse("data").toarray()).float()
    
    # Set model hyperparameters.
    k = 100
    
    # Set optimization hyperparameters.
    lr = 0.1
    num_epoch = 100
    lamb = 0.01
    print(train_data.shape[1], len(subjects))
    model = NN_model(train_data.shape[1], len(subjects), k)
    train_b(model, lr, lamb, train_data, train_sparse, zero_train_matrix, valid_data, num_epoch)
    
    # test_acc = evaluate(model, zero_train_matrix, test_data)
    # print("Test Accuracy: {}".format(test_acc))
    
    # for lamb in [0.001, 0.01, 0.1, 1]:
    #     model = NN_model(train_matrix.shape[1], num_epoch)
    #     train_b(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, k)
    #     test_acc = evaluate(model, zero_train_matrix, test_data)
    #     print("Test Accuracy: {}".format(test_acc))

if __name__ == "__main__":
    main()
