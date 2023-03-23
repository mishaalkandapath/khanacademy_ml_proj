from utils_a import *

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A sparse matrix of all users, questions and with value of 1, 0, 
        or nan based on correctness
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # stretches beta and theta vectors into matrices that are the size of data
    # so that numpy operations can be done
    beta_stretch = np.vstack([beta]*data.shape[0])
    theta_stretch = np.column_stack([theta]*data.shape[1])

    # term1 = log(p(c_ij = 1 | theta_i, beta_j))
    term1 = np.power(np.log(np.maximum(.0000001, sigmoid(theta_stretch - beta_stretch))), data)
    term1[np.isnan(term1)] = 0
    
    # term2 = log(p(c_ij = 0 | theta_i, beta_j))
    term2 = np.power(1 - np.log(np.maximum(.0000001, sigmoid(theta_stretch - beta_stretch))), 1-data)
    term2[np.isnan(term2)] = 0
    
    # sum all the terms calculated above
    log_lklihood = np.sum(term1*term2)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A sparse matrix of all users, questions and with value of 1, 0, 
        or nan based on correctness
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # stretch theta and beta vectors to ease numpy operations
    theta_stretch = np.column_stack([theta]*len(beta))  # each column is theta
    beta_stretch = np.vstack([beta]*len(theta))  # each row is beta

    # these are derived from derivative formulas
    theta_is_correct = 1/(1+np.exp(theta_stretch - beta_stretch))  # d/dtheta ln(p(c_ij = 1 | theta_i, beta_j))
    beta_is_correct = -theta_is_correct   # happens to equal negative of theta partial --  d/dbeta ln(p(c_ij = 1 | theta_i, beta_j))
    
    theta_not_correct = -theta_is_correct  # fill matrix with updates for if c_ij = 0
    # mask using sparse matrix to indicate what entries are correct or not
    theta_not_correct = np.where(data == 0, theta_not_correct, 0)
    theta_is_correct = np.where(data == 1, theta_is_correct, 0)
    
    beta_not_correct = -beta_is_correct  # fill matrix with updates for if c_ij = 0
    # mask using sparse matrix to indicate what entries are correct or not
    beta_not_correct = np.where(data == 0, beta_not_correct, 0)
    beta_is_correct = np.where(data == 1, beta_is_correct, 0)
    
    # sum and multiply by learning rate
    theta += lr*np.sum(theta_is_correct + theta_not_correct, axis=1)
    beta += lr*np.sum((beta_is_correct + beta_not_correct).T, axis=1)
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    # initializing weights to 0.5 - assuming as little as possible
    theta = np.ones(data.shape[0])*0.5
    beta = np.ones(data.shape[1])*0.5

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    theta, beta, val_acc = irt(sparse_matrix, val_data, .1, 100);
    print("Final validation accuracy: " + str(val_acc[len(val_data) - 1]))
    print("Final testing accuracy: " + str(evaluate(test_data, theta, beta)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    j = [1, 2, 3]
    prob_correct = []
    # plot the probability over theta
    for i in j:
        prob_correct.append(sigmoid(theta - beta[i]))
        
    plt.plot(theta, prob_correct[1], label="one")
    plt.plot(theta, prob_correct[2], label="two")
    plt.xlabel("Theta")
    plt.ylabel("Probability")
    plt.legend(loc="upper left")
    plt.savefig("item_response.png")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
