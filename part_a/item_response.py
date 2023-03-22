from utils import *

import numpy as np
import math


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # data_arr = np.column_stack(data["user_id"], data["question_id"], data["is_correct"])
    # log_lklihood = np.sum(np.sum(np.power(sigmoid(theta - beta), data)*
    #                        np.power(1-sigmoid(beta - beta), data)))
    
    
    # from data dict- order theta and beta according to the order of the data (theta[0] = first user in data_dict)
    theta_map = []
    beta_map = []
    for a in range(len(data['is_correct'])):
        if not(data['user_id'][a] in theta_map):
            theta_map.append(data['user_id'][a])
        if not(data['question_id'][a] in beta_map):
            beta_map.append(data['question_id'][a])
    
    log_lklihood = 0.
    for a in range(len(data['is_correct'])):
        cur_prob = math.log(sigmoid(theta[theta_map.index(data['user_id'][a])] - beta[beta_map.index(data['question_id'][a])]))
        if data['is_correct'][a] == 1:
            log_lklihood += cur_prob
        else:
            log_lklihood += 1 - cur_prob
    
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

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    theta_map = []
    beta_map = []
    for a in range(len(data['is_correct'])):
        if not(data['user_id'][a] in theta_map):
            theta_map.append(data['user_id'][a])
        if not(data['question_id'][a] in beta_map):
            beta_map.append(data['question_id'][a])
    data_arr = np.column_stack((data["user_id"], data["question_id"], data["is_correct"]))
    copy_theta = theta.copy()
    
    for i in range(len(theta)):
        # print(theta[i] - beta)
        # t_update = lr * np.sum(1/(1 + np.exp(theta[i] - beta)))
        t_update = 0
        for j in range(len(beta)):
            prob_if_correct = 1/(1 + np.exp(theta[i] - beta[j]))
            index = -1
            masked = np.copy(data_arr)
            # masked[np.where((masked[:,0] == theta_map[i]) & (masked[:,1] == beta_map[j]))]
            masked_1 = masked[:, 0]
            masked_2 = masked[:, 1]
            masked_1 = np.where(masked_1 == theta_map[i], 1, 0)
            masked_2 = np.where(masked_2 == beta_map[j], 1, 0)
            masked_end = masked_1 & masked_2
            masked_3 = masked[:, 2]
            masked = masked_3 & masked_end
            # print(masked, theta_map[i], beta_map[j])
            if np.sum(masked) == 1:
                t_update += prob_if_correct
            else:
                t_update -= prob_if_correct
        theta[i] += lr*t_update
    for j in range(len(beta)):
        # print(copy_theta - beta[j])
        # beta[j] += lr * np.sum(1/(1 + np.exp(copy_theta - beta[j])))
        b_update = 0
        for i in range(len(theta)):
            prob_if_correct = 1/(1 + np.exp(theta[i] - beta[j]))
            index = -1
            for aa in range(len(data['is_correct'])):
                if data['user_id'][aa] == theta[i] and data['user_id'][aa] == beta[j]:
                    index = aa
                    break
            if data['is_correct'][index] == 1:
                b_update += prob_if_correct
            else:
                b_update -= prob_if_correct
        # print(t_update)
        beta[i] += lr*b_update
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
    theta_map = set()
    beta_map = set()
    for a in range(len(data['is_correct'])):
        theta_map.add(data['user_id'][a])
        beta_map.add(data['question_id'][a])
    theta = np.ones(len(theta_map))*0.5
    beta = np.ones(len(beta_map))*0.5

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print(theta[0])
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
    sparse_matrix = load_train_sparse("data")
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    irt(train_data, val_data, 1, 3);
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
