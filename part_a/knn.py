from sklearn.impute import KNNImputer
from utils import *

import matplotlib.pyplot as plt
import numpy as np

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    imputer = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = imputer.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("data").toarray()
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)
    
    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################    
    k = [1, 6, 11, 16, 21, 26]
    user_k = (-1, -1)
    item_k = (-1, -1)
    val_acc_u = []
    val_acc_i = []
    for cur_k in k:
        cur_acc_u = knn_impute_by_user(sparse_matrix, val_data, cur_k)
        cur_acc_i = knn_impute_by_item(sparse_matrix, val_data, cur_k)
        val_acc_u.append(cur_acc_u)
        val_acc_i.append(cur_acc_i)
        if user_k[0] < cur_acc_u:
            user_k = (cur_acc_u, cur_k)
        if user_k[0] < cur_acc_i:
            user_k = (cur_acc_i, cur_k)
    knn_impute_by_user(sparse_matrix, test_data, user_k[1])
    knn_impute_by_item(sparse_matrix, test_data, user_k[1])
    plt.plot(k, val_acc_u, label="User")
    plt.plot(k, val_acc_i, label="Question")
    plt.xlabel("k value")
    plt.ylabel("Validation Accuracy")
    plt.legend(loc="upper left")
    plt.savefig("knn.png")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
