from sklearn.impute import KNNImputer
from utils_a import *

import matplotlib.pyplot as plt

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


def knn_impute_ensemble(matrix, valid_data, k):
    """Returns just the trained matrix without calculating accuracy- useful for ensemble

    Args:
        matrix (2D sparse matrix): training data matrix (empty entries are filled)
        valid_data (dict): valid data used to evaluate parameter k
        k (int): number of nearest neighbors

    Returns:
        sparse matrix: filled training data matrix according to knn
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return mat


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
    # apply imputer using transpose- i.e. items = data points, users = features
    mat = imputer.fit_transform(matrix.T)  
    acc = sparse_matrix_evaluate(valid_data, mat.T)  # validate with transpose so shape matches
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
    user_k = (-1, -1)   # stores k with max score and its score
    item_k = (-1, -1)   # stores k with max score and its score
    val_acc_u = []
    val_acc_i = []
    # loops through k values for each comparison
    for cur_k in k:
        # fill matrix and calculate accuracy using helper function
        cur_acc_u = knn_impute_by_user(sparse_matrix, val_data, cur_k)
        cur_acc_i = knn_impute_by_item(sparse_matrix, val_data, cur_k)
        val_acc_u.append(cur_acc_u)
        val_acc_i.append(cur_acc_i)
        if user_k[0] < cur_acc_u:
            user_k = (cur_acc_u, cur_k)
        if item_k[0] < cur_acc_i:
            item_k = (cur_acc_i, cur_k)
    # report accuracy for the best performing k for item and user
    knn_impute_by_user(sparse_matrix, test_data, user_k[1])
    knn_impute_by_item(sparse_matrix, test_data, user_k[1])
    
    # plot results
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
