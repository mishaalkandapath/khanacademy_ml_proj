from sklearn.cluster import KMeans, AgglomerativeClustering
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.sparse import csr_matrix, save_npz
from scipy.spatial.distance import cosine

import time
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor

dtype = torch.float64
device_id = "cpu"


subjects = []

def load_subjects():
    """Load all subject ids""" 
    global subjects
    with open("data/subject_meta.csv") as f:
        for line in f:
            line_c = line.strip().split(",")
            if line_c[0] != "subject_id":
                title = line.replace(line_c[0] + ",", "")
                title = title.replace("\n", "")
                title = title.replace('-Others', "")
                if line_c[1][0] == '"':
                    title = title[1:-1]
                    title = title.replace(",", "")
                subjects.append((title))

# def load_distance_matrix():
    

#     # normalize all word distances to be between 0 - 1
#     distance_matrix = np.zeros((len(subjects), len(subjects)))
#     for a in subjects:
#         a_dist = []
#         for b in subjects:
#             if distance_matrix[b][a] == 0:
#                 if a != b:
#                     a = a.split(" ")
#                     b = b.split(" ")
#                     a_dist.append(word_vectors.wmdistance(a, b))
#                     print(wmd_distance)
#                 else:
#                     a_dist.append(0)
#             else:
#                 a_dist.append(distance_matrix[b][a])
#         distance_matrix[subjects.index(a)] = a_dist
    
#     scalar = MinMaxScaler()
#     distance_matrix_normalized = scalar.fit_transform(distance_matrix)
#     dist_mat = csr_matrix(distance_matrix_normalized)
#     save_npz("data/subj_dist.npz", dist_mat)
#     # hopefully this matrix is loaded in so this expensive calculation doesn't have to be done again
#     return distance_matrix_normalized

def KMeans(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c


def main():
    load_subjects()
    # dist_mat = load_distance_matrix()

    # k = [3, 4, 5, 7, 10]
    # clusters = []
    # for cur_k in k:
    #     clusters.append(AgglomerativeClustering(n_clusters=cur_k, metric="precomputed", linkage="complete", compute_distances=True).fit_predict(dist_mat))
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(subjects)
    words = vectorizer.get_feature_names_out()
    vectors = vectorizer.fit(subjects)
    vectorizer = CountVectorizer(binary=True, vocabulary=words)
    vectorizer.fit_transform(subjects)
    binary_vectors = np.asarray(vectorizer.transform(subjects).toarray())
    binary_vectors = torch.from_numpy(binary_vectors)
    binary_vectors = binary_vectors.float()
    print(binary_vectors.shape)
    print(binary_vectors[0])
    # # calculate cosine similarity
    # cos_sim = dot(binary_vectors[0], binary_vectors[1])/(norm(binary_vectors[0])*norm(binary_vectors[1]))
    # cos_sim1 = dot(binary_vectors[20], binary_vectors[31])/(norm(binary_vectors[20])*norm(binary_vectors[31]))
    # print(cos_sim, cos_sim1)
    # enc = OneHotEncoder(handle_unknown='ignore')
    # enc.fit_transform(words)

    variances = []
    for i in range(100):
        cl, c = KMeans(binary_vectors, K=10, Niter=10, verbose=True)
        #find the variance in c:
        variances.append(np.var(cl.numpy(), axis=0))
        print(variances[-1])
    #plot this
    plt.plot(variances)
    plt.show()




    

    
if __name__ == "__main__":
    main()