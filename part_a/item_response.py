from utils_a import *

import numpy as np
import matplotlib.pyplot as plt
import math
alternate = True

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))

groups = {1: [1, 23, 43, 51, 64, 76, 87, 92, 107, 124, 134, 146, 156, 174, 188, 200, 211, 223, 238, 247, 266, 277, 289, 296, 303, 312, 328, 337, 348, 354, 361, 374, 385], 
          2: [4, 19, 36, 41, 55, 62, 71, 78, 85, 99, 113, 126, 141, 153, 170, 183, 193, 207, 219, 235, 244, 260, 272, 283, 293, 300, 309, 325, 335, 345, 352, 359, 370, 381], 
          3: [10, 30, 46, 59, 67, 81, 96, 104, 118, 129, 138, 151, 162, 177, 192, 203, 216, 228, 242, 251, 268, 279, 292, 299, 308, 318, 332, 341, 351, 358, 365, 377, 388], 
          4: [6, 25, 42, 57, 63, 77, 89, 94, 109, 125, 136, 149, 159, 171, 185, 196, 209, 221, 234, 245, 261, 274, 287, 295, 305, 315, 330, 340, 349, 356, 363, 373, 383], 
          5: [3, 17, 34, 48, 54, 70, 83, 91, 105, 120, 132, 144, 154, 167, 181, 190, 204, 217, 230, 241, 256, 269, 282, 290, 298, 307, 320, 334, 343, 353, 360, 372, 384], 
          6: [8, 22, 40, 52, 66, 75, 86, 97, 111, 123, 133, 145, 157, 173, 187, 198, 210, 225, 239, 249, 263, 275, 286, 295, 302, 314, 327, 338, 347, 355, 366, 375, 386], 
          7: [12, 28, 45, 60, 68, 82, 93, 103, 117, 127, 140, 152, 165, 178, 191, 202, 214, 226, 243, 255, 270, 280, 291, 301, 316, 324, 339, 350, 357, 368, 379, 390], 
          8: [2, 16, 31, 47, 53, 69], 
          9: [103, 105, 107, 113, 117, 118, 119, 120, 122, 123, 126, 128, 129, 131, 133], 
          10: [34, 50, 51, 55, 60, 68, 69, 70, 72, 74, 76, 81, 84, 85, 87, 88, 91, 92, 94, 97, 99], 
          11: [16, 25, 26, 27, 28, 29, 30, 31, 32, 35, 39, 42, 43, 46, 47, 49, 56, 57, 59, 61, 62, 63, 65, 66, 71, 73, 77, 78, 79, 82, 83, 89, 95, 96, 98], 
          12: [21, 22, 33, 38, 44, 45, 48, 52, 54, 58, 64, 75, 80, 86, 90, 93], 
          13: [2, 5, 10, 14, 17, 20, 23, 24, 37, 40, 41, 67, 100, 101, 102, 104, 106, 108, 110, 111, 112, 114, 115, 116, 121, 124, 125, 127, 130, 132, 134, 135], 
          14: [0, 9, 11, 160, 161, 163, 164, 166, 186, 199, 201, 205, 206, 224, 262, 273, 329, 331, 13, 158, 189, 213, 215, 218, 220, 222, 227, 231, 246, 248, 250, 252, 253, 254, 264, 267, 271, 276, 278, 281, 284, 285], 
          15: [176, 208, 212, 229, 257, 258, 259, 265, 7, 15, 18, 168, 169, 172, 175, 179, 180, 182, 184, 194, 195, 197], 
          16: [139, 142, 143, 232, 233, 236, 237, 240, 378, 382], 
          17: [288, 294, 297, 304, 306, 310, 311, 313, 336, 344, 346, 362, 364, 367, 369, 371, 376, 380, 387], 
          18: [317, 319, 321, 322, 323, 326, 333, 342, 137, 147, 148, 150, 155]}

# def load_subjects():
#     """Load all subject ids""" 
#     subjects = {}
#     excluded = []
#     global grouped
#     global second
#     global third
#     grouped = set(grouped)
#     with open("data/subject_meta.csv") as f:
#         for line in f:
#             line1 = line.strip().split(",")
#             if line1[0] != "subject_id":
#                 line = line.replace(line1[0], "")
#                 line = line.replace('\n', "")
#                 line = line.replace('\r', "")
#                 line = line.replace(',', "")
#                 if line[0] == '"':
#                     line = line[1:-1]
#                 subjects[int(line1[0])] = line
    # for subj_id in grouped:
    #     if subj_id not in subjects: 
    #         excluded.append(subj_id)
    #     else:
    #         subjects.pop(subj_id)
    # for subj in second:
    #     if subj not in subjects: 
    #         excluded.append(subj)
    #     else:
    #         subjects.pop(subj)
    # for subj in third:
    #     if subj not in subjects: 
    #         excluded.append(subj)
    #     else:
    #         subjects.pop(subj)
    # return subjects, excluded

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
    return log_lklihood

def log_likelihood_dict(data, theta, beta):
    log_lklihood = 0
    for i, q in enumerate(data["question_id"]):
        c = data["is_correct"][i]
        u = data["user_id"][i]
        if c > 0:
            log_lklihood += math.log(max(.0000001, sigmoid(theta[u] - beta[q])))
        else:
            log_lklihood += math.log(1 - max(.0000001, sigmoid(theta[u] - beta[q])))
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
    global alternate
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
    if alternate:
        theta += lr*np.sum(theta_is_correct + theta_not_correct, axis=1)
    else:
        beta += lr*np.sum((beta_is_correct + beta_not_correct).T, axis=1)
    alternate = not alternate
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
    val_lg_likelihood = []
    train_lg_likelihood = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        train_lg_likelihood.append(neg_lld)
        val_lg_likelihood.append(log_likelihood_dict(val_data, theta=theta, beta=beta))
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_lg_likelihood, val_lg_likelihood


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
    iterations = 30
    theta, beta, val_acc, train_lg, val_lg = irt(sparse_matrix, val_data, .05, iterations)
    print("Final validation accuracy: " + str(val_acc[len(val_acc) - 1]))
    print("Final testing accuracy: " + str(evaluate(test_data, theta, beta)))
    # print([x for x in range(1, iterations + 1)])
    plt.plot([x for x in range(1, iterations + 1)], train_lg, label="Train Log Likelihood")
    plt.plot([x for x in range(1, iterations + 1)], val_lg, label="Validation Log Likelihood")
    plt.xlabel("Iterations")
    plt.ylabel("Log Likelihood")
    plt.legend(loc="upper left")
    plt.savefig("item_response.png")    
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
    theta_sorted = np.sort(theta)
    for i in j:
        prob_correct.append(sigmoid(theta_sorted - beta[i]))
        
    plt.plot(theta, prob_correct[0], label="one")
    plt.plot(theta, prob_correct[1], label="two")
    plt.xlabel("Theta")
    plt.ylabel("Probability")
    plt.legend(loc="upper left")
    plt.savefig("irt_d.png")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
