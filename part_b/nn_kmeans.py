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
from sklearn.metrics.pairwise import pairwise_distances

#storing clusters globally
subjects = []
q_clusters_15subjects = [30, 33, 12, 19, 7, 7, 13, 35, 2, 18, 34, 13, 5, 16, 6, 2, 31, 9, 25, 35, 32, 7, 2, 9, 22, 21, 30, 10, 5, 7, 9, 16, 12, 8, 2, 6, 23, 6, 12, 9, 25, 20, 22, 35, 12, 15, 27, 2, 28, 16, 8, 27, 12, 10, 13, 4, 8, 9, 15, 28, 23, 11, 1, 20, 12, 18, 1, 12, 18, 3, 19, 3, 33, 21, 7, 7, 11, 3, 9, 28, 5, 8, 34, 16, 27, 19, 2, 7, 34, 15, 8, 1, 26, 14, 11, 12, 12, 13, 15, 12, 30, 17, 13, 30, 7, 7, 3, 2, 2, 34, 27, 24, 7, 1, 33, 3, 0, 28, 9, 13, 19, 5, 32, 30, 14, 24, 13, 16, 6, 12, 7, 12, 6, 35, 1, 25, 27, 1, 14, 3, 22, 2, 13, 27, 2, 17, 13, 33, 33, 8, 34, 2, 19, 12, 19, 20, 10, 4, 7, 1, 13, 13, 15, 8, 5, 0, 0, 1, 6, 21, 12, 15, 2, 8, 10, 2, 20, 18, 11, 21, 24, 18, 21, 13, 15, 8, 35, 35, 1, 2, 9, 27, 35, 12, 9, 25, 34, 7, 12, 3, 5, 3, 2, 13, 27, 12, 5, 29, 7, 24, 7, 9, 1, 9, 16, 0, 23, 22, 19, 3, 9, 8, 3, 20, 9, 20, 12, 12, 30, 22, 2, 18, 9, 13, 0, 21, 15, 4, 3, 15, 27, 22, 23, 1, 2, 22, 3, 22, 9, 28, 17, 5, 9, 9, 27, 22, 19, 12, 1, 32, 18, 12, 23, 4, 10, 31, 13, 3, 16, 13, 19, 6, 27, 27, 25, 8, 15, 9, 12, 3, 6, 19, 33, 7, 7, 7, 34, 7, 3, 16, 25, 32, 24, 34, 28, 0, 34, 12, 3, 21, 6, 24, 7, 7, 3, 7, 12, 4, 17, 8, 5, 8, 7, 17, 24, 23, 25, 12, 2, 35, 12, 31, 20, 28, 5, 1, 25, 13, 16, 1, 29, 1, 17, 16, 27, 2, 8, 12, 12, 35, 3, 7, 27, 16, 13, 24, 3, 12, 2, 4, 3, 0, 21, 4, 2, 33, 27, 2, 16, 13, 11, 7, 12, 20, 2, 22, 12, 0, 16, 12, 1, 28, 12, 15, 2, 18, 1, 23, 21, 15, 18, 11, 11, 11, 28, 33, 0, 13, 32, 8, 1, 21, 27, 0, 4, 5, 17, 7, 13, 19, 9, 0, 13, 2, 28, 2, 16, 14, 30, 6, 15, 12, 19, 16, 9, 13, 20, 9, 1, 32, 5, 7, 2, 1, 35, 12, 28, 24, 16, 32, 35, 21, 5, 20, 1, 23, 8, 15, 8, 30, 19, 31, 16, 35, 3, 16, 4, 15, 11, 3, 16, 3, 9, 7, 1, 0, 27, 23, 5, 3, 7, 8, 14, 19, 2, 32, 29, 27, 12, 2, 8, 9, 26, 13, 23, 12, 10, 2, 5, 22, 20, 31, 24, 9, 25, 28, 33, 29, 18, 0, 7, 12, 29, 18, 6, 8, 1, 12, 16, 30, 1, 23, 8, 12, 7, 21, 19, 12, 35, 12, 8, 30, 15, 32, 20, 15, 2, 6, 1, 0, 7, 7, 28, 22, 13, 29, 0, 15, 12, 19, 25, 24, 22, 19, 8, 12, 16, 25, 9, 1, 7, 5, 22, 5, 5, 23, 5, 8, 13, 27, 13, 26, 6, 5, 12, 20, 16, 22, 11, 16, 8, 12, 15, 1, 6, 30, 2, 27, 1, 34, 6, 1, 28, 21, 11, 23, 27, 1, 12, 8, 10, 2, 12, 16, 12, 5, 20, 12, 34, 15, 7, 26, 1, 13, 13, 13, 27, 30, 5, 7, 27, 9, 9, 8, 6, 29, 5, 8, 28, 15, 23, 30, 9, 27, 25, 8, 15, 2, 27, 5, 23, 19, 2, 31, 19, 12, 5, 11, 5, 5, 19, 11, 15, 2, 13, 9, 28, 12, 1, 12, 0, 13, 12, 0, 32, 9, 21, 9, 19, 17, 15, 35, 13, 18, 13, 12, 4, 31, 23, 7, 21, 5, 1, 2, 34, 13, 1, 10, 1, 12, 0, 1, 21, 9, 16, 33, 8, 18, 0, 18, 23, 6, 14, 34, 21, 5, 6, 9, 1, 16, 20, 24, 19, 31, 12, 9, 2, 24, 30, 11, 5, 12, 33, 0, 11, 18, 4, 28, 15, 27, 30, 19, 16, 9, 14, 27, 12, 18, 29, 5, 8, 24, 0, 21, 6, 31, 21, 5, 5, 13, 0, 12, 10, 5, 7, 2, 11, 13, 21, 12, 1, 20, 18, 6, 1, 18, 27, 5, 22, 7, 9, 35, 32, 13, 5, 12, 30, 7, 10, 7, 20, 14, 19, 3, 15, 16, 12, 13, 30, 19, 18, 2, 7, 2, 23, 16, 22, 26, 29, 13, 28, 15, 8, 12, 7, 28, 27, 16, 29, 34, 19, 5, 21, 13, 2, 27, 12, 7, 33, 16, 3, 0, 18, 9, 23, 12, 24, 17, 21, 1, 8, 18, 2, 7, 0, 4, 30, 21, 1, 6, 16, 0, 6, 13, 31, 12, 7, 15, 11, 17, 5, 8, 1, 7, 11, 31, 20, 16, 6, 7, 20, 20, 7, 7, 12, 13, 5, 12, 2, 15, 31, 28, 27, 6, 13, 0, 1, 11, 20, 22, 8, 6, 1, 32, 7, 12, 16, 13, 5, 34, 35, 22, 12, 2, 19, 6, 1, 16, 3, 10, 28, 32, 23, 7, 6, 11, 13, 32, 4, 25, 22, 27, 1, 5, 17, 2, 27, 6, 12, 3, 15, 13, 12, 5, 30, 5, 8, 8, 21, 1, 8, 2, 15, 25, 25, 7, 0, 20, 12, 27, 22, 23, 1, 13, 5, 27, 12, 19, 10, 5, 35, 27, 10, 20, 17, 25, 18, 32, 15, 7, 8, 16, 15, 13, 2, 6, 0, 19, 8, 7, 9, 21, 2, 23, 7, 7, 13, 18, 30, 35, 12, 35, 0, 3, 17, 3, 12, 19, 0, 30, 15, 30, 7, 15, 13, 1, 29, 29, 22, 19, 32, 2, 29, 1, 1, 20, 34, 2, 1, 23, 4, 11, 5, 30, 12, 19, 7, 18, 1, 8, 13, 0, 28, 17, 11, 9, 19, 13, 5, 30, 22, 7, 10, 13, 11, 30, 2, 6, 2, 0, 7, 23, 19, 2, 18, 5, 9, 8, 22, 12, 22, 13, 2, 12, 20, 9, 6, 30, 13, 12, 29, 30, 1, 8, 4, 35, 16, 11, 6, 3, 30, 9, 17, 5, 24, 18, 28, 8, 23, 23, 7, 27, 0, 18, 27, 16, 30, 28, 12, 15, 8, 21, 1, 11, 2, 5, 19, 5, 12, 23, 26, 14, 27, 30, 30, 19, 6, 21, 5, 23, 15, 7, 12, 24, 29, 3, 7, 18, 11, 30, 19, 5, 4, 12, 12, 22, 13, 23, 20, 13, 2, 3, 8, 18, 12, 25, 35, 28, 4, 31, 7, 17, 6, 28, 8, 7, 11, 3, 5, 0, 19, 30, 7, 12, 28, 16, 7, 8, 5, 25, 24, 15, 19, 2, 5, 35, 7, 29, 33, 27, 25, 22, 11, 24, 20, 16, 6, 0, 27, 4, 19, 11, 32, 11, 7, 22, 11, 12, 32, 2, 1, 17, 20, 24, 31, 5, 13, 14, 7, 19, 23, 3, 33, 10, 20, 2, 23, 8, 0, 2, 7, 33, 3, 7, 23, 21, 16, 13, 22, 23, 30, 0, 17, 33, 12, 12, 13, 2, 23, 7, 7, 8, 7, 22, 12, 6, 14, 22, 4, 8, 6, 27, 0, 27, 5, 24, 13, 0, 1, 12, 7, 5, 2, 24, 6, 5, 23, 1, 5, 17, 16, 33, 5, 21, 10, 7, 5, 8, 16, 5, 34, 6, 28, 27, 2, 7, 2, 27, 21, 27, 7, 35, 7, 33, 0, 7, 23, 7, 27, 23, 9, 10, 1, 1, 26, 21, 3, 19, 17, 1, 14, 14, 23, 11, 13, 18, 5, 17, 24, 25, 7, 2, 13, 25, 9, 14, 27, 2, 30, 6, 2, 31, 27, 17, 8, 12, 3, 0, 1, 9, 12, 5, 6, 8, 21, 21, 7, 13, 17, 3, 21, 12, 12, 22, 28, 21, 12, 5, 9, 0, 12, 30, 15, 3, 7, 8, 16, 2, 23, 15, 31, 3, 16, 28, 12, 14, 11, 7, 21, 15, 30, 24, 35, 6, 17, 5, 29, 2, 13, 1, 12, 6, 19, 13, 13, 30, 13, 12, 28, 16, 1, 24, 28, 16, 3, 22, 19, 35, 12, 30, 1, 20, 35, 13, 15, 15, 21, 14, 16, 21, 24, 9, 12, 13, 27, 5, 1, 7, 35, 13, 12, 8, 7, 18, 11, 15, 35, 2, 23, 8, 18, 9, 6, 12, 15, 16, 26, 24, 19, 32, 27, 23, 12, 15, 19, 8, 16, 12, 5, 9, 10, 27, 3, 27, 12, 1, 2, 2, 15, 34, 17, 2, 13, 13, 5, 30, 27, 3, 19, 3, 13, 27, 12, 25, 15, 19, 12, 11, 0, 18, 7, 8, 15, 9, 13, 13, 19, 12, 11, 23, 5, 12, 8, 12, 12, 12, 30, 6, 2, 28, 33, 4, 12, 12, 32, 3, 17, 11, 10, 12, 12, 28, 20, 12, 13, 23, 15, 19, 23, 15, 29, 27, 23, 5, 13, 28, 3, 9, 27, 26, 20, 24, 32, 30, 21, 15, 34, 23, 0, 7, 5, 2, 13, 1, 13, 30, 2, 8, 22, 30, 13, 5, 9, 17, 35, 1, 1, 21, 12, 12, 0, 13, 12, 7, 0, 14, 34, 19, 13, 6, 15, 14, 13, 7, 4, 19, 13, 21, 28, 15, 12, 9, 22, 4, 23, 2, 22, 2, 21, 20, 23, 27, 28, 19, 30, 12, 8, 3, 27, 32, 0, 5, 28, 19, 18, 7, 12, 13, 20, 12, 27, 24, 2, 3, 3, 9, 32, 1, 2, 22, 9, 31, 16, 2, 9, 28, 22, 20, 9, 29, 9, 2, 1, 13, 1, 7, 1, 28, 20, 24, 13, 20, 30, 10, 11, 27, 2, 34, 22, 13, 34, 13, 33, 5, 22, 15, 17, 9, 13, 19, 28, 16, 1, 12, 23, 0, 9, 31, 9, 9, 8, 21, 30, 10, 31, 33, 5, 9, 5, 20, 31, 13, 32, 32, 13, 6, 0, 9, 35, 1, 15, 12, 20, 12, 21, 12, 2, 7, 23, 6, 15, 25, 3, 24, 22, 9, 28, 7, 18, 29, 13, 1, 7, 5, 35, 7, 22, 19, 0, 6, 2, 33, 16, 1, 20, 27, 1, 2, 18, 27, 22, 8, 13, 27, 31, 1, 22, 3, 23, 0, 35, 29, 2, 35, 22, 18, 20, 13, 8, 13, 3, 34, 15, 7, 5, 30, 27, 0, 14, 12, 12, 19, 20, 30, 17, 33, 7, 2, 16, 14, 16, 7, 13, 27, 31, 9, 30, 0, 33, 7, 21, 18, 8, 5, 13, 6, 8, 5, 22, 12, 12, 23, 7, 12]
q_clusters_30subjects = [25, 35, 1, 40, 27, 27, 39, 12, 1, 2, 51, 34, 41, 6, 50, 1, 7, 48, 33, 10, 30, 32, 7, 42, 4, 35, 4, 49, 13, 27, 46, 28, 7, 43, 1, 8, 27, 50, 7, 36, 33, 15, 11, 12, 7, 23, 38, 1, 51, 28, 43, 4, 1, 30, 31, 18, 10, 36, 35, 9, 27, 47, 37, 15, 21, 19, 27, 1, 2, 36, 40, 36, 0, 53, 53, 53, 17, 25, 6, 31, 13, 26, 48, 44, 6, 6, 1, 14, 51, 48, 43, 16, 32, 11, 30, 7, 3, 39, 7, 7, 26, 2, 29, 26, 32, 53, 3, 1, 1, 42, 4, 44, 5, 16, 0, 3, 29, 4, 6, 31, 6, 13, 30, 25, 0, 44, 34, 28, 8, 7, 5, 21, 50, 12, 37, 33, 38, 16, 11, 24, 46, 1, 29, 23, 7, 2, 31, 35, 0, 43, 51, 21, 40, 7, 40, 15, 30, 6, 27, 37, 31, 31, 23, 43, 21, 4, 52, 37, 8, 22, 7, 23, 20, 26, 49, 14, 15, 19, 17, 22, 21, 2, 22, 29, 35, 2, 12, 12, 37, 1, 31, 38, 12, 1, 46, 33, 51, 53, 7, 36, 13, 3, 1, 31, 23, 41, 13, 13, 32, 44, 32, 36, 37, 46, 6, 2, 9, 11, 50, 36, 6, 10, 36, 15, 6, 15, 21, 21, 25, 46, 1, 2, 6, 34, 4, 22, 48, 18, 36, 35, 38, 4, 5, 16, 1, 11, 36, 4, 22, 9, 2, 41, 36, 46, 38, 48, 40, 9, 16, 30, 19, 1, 5, 18, 49, 7, 20, 36, 28, 20, 40, 50, 23, 38, 33, 26, 23, 6, 21, 24, 50, 40, 0, 53, 5, 27, 51, 27, 24, 6, 33, 30, 44, 51, 5, 29, 46, 1, 3, 22, 15, 44, 53, 5, 36, 53, 21, 18, 2, 2, 13, 43, 5, 2, 44, 27, 33, 3, 1, 12, 1, 7, 15, 9, 13, 27, 33, 34, 6, 37, 13, 37, 2, 6, 38, 1, 43, 1, 3, 12, 25, 27, 6, 28, 29, 34, 21, 7, 1, 18, 3, 29, 5, 6, 1, 0, 38, 1, 44, 29, 17, 5, 21, 15, 7, 4, 7, 29, 6, 3, 27, 4, 7, 35, 21, 2, 37, 45, 9, 23, 37, 17, 47, 17, 9, 32, 29, 34, 49, 26, 16, 22, 23, 52, 18, 13, 19, 19, 53, 40, 6, 29, 31, 1, 9, 1, 8, 11, 25, 8, 48, 7, 6, 28, 6, 20, 15, 6, 16, 49, 41, 27, 1, 16, 10, 7, 4, 21, 8, 30, 10, 22, 13, 15, 37, 5, 2, 23, 43, 26, 40, 21, 6, 12, 24, 28, 18, 48, 17, 36, 42, 3, 6, 9, 5, 52, 37, 45, 41, 21, 27, 43, 11, 40, 7, 30, 13, 37, 7, 1, 43, 6, 27, 20, 5, 7, 27, 1, 13, 4, 15, 19, 21, 36, 33, 27, 0, 36, 2, 4, 5, 20, 13, 37, 8, 10, 16, 7, 28, 4, 37, 27, 43, 7, 53, 22, 50, 21, 10, 21, 2, 4, 48, 50, 15, 23, 1, 8, 37, 29, 27, 14, 4, 11, 50, 13, 4, 23, 3, 40, 33, 34, 11, 40, 26, 7, 44, 33, 36, 16, 27, 13, 48, 13, 13, 45, 13, 2, 34, 38, 29, 32, 8, 13, 21, 15, 44, 11, 47, 28, 10, 48, 48, 10, 8, 25, 1, 4, 37, 51, 8, 37, 14, 22, 17, 5, 4, 37, 7, 43, 49, 1, 23, 6, 7, 13, 15, 7, 51, 35, 14, 32, 37, 25, 31, 29, 38, 41, 13, 32, 38, 6, 42, 10, 50, 24, 13, 43, 9, 35, 45, 26, 6, 4, 33, 43, 48, 1, 4, 13, 45, 50, 7, 7, 6, 7, 13, 47, 13, 13, 40, 17, 23, 1, 20, 6, 4, 7, 16, 7, 4, 39, 7, 29, 30, 6, 5, 36, 40, 2, 23, 12, 34, 26, 39, 29, 18, 19, 27, 5, 22, 21, 16, 1, 51, 34, 37, 49, 37, 7, 52, 16, 53, 6, 44, 0, 2, 26, 29, 19, 9, 50, 11, 51, 35, 41, 8, 46, 37, 44, 15, 21, 40, 19, 7, 36, 1, 34, 4, 17, 13, 21, 0, 52, 47, 37, 18, 9, 23, 37, 41, 50, 6, 46, 11, 38, 7, 37, 13, 41, 43, 21, 29, 35, 50, 19, 22, 41, 13, 34, 4, 7, 49, 13, 53, 1, 17, 31, 2, 29, 16, 15, 19, 50, 37, 26, 38, 41, 4, 32, 10, 12, 50, 20, 13, 7, 25, 27, 49, 53, 15, 11, 50, 36, 48, 28, 7, 39, 25, 40, 2, 1, 53, 1, 5, 28, 4, 31, 13, 20, 27, 23, 43, 21, 32, 9, 3, 6, 36, 46, 40, 13, 53, 20, 1, 4, 1, 27, 0, 28, 1, 29, 13, 6, 45, 7, 44, 2, 35, 5, 43, 2, 1, 27, 52, 18, 25, 22, 37, 50, 44, 4, 8, 20, 7, 7, 5, 23, 17, 2, 21, 43, 16, 27, 47, 7, 15, 28, 50, 27, 15, 15, 32, 5, 7, 34, 41, 7, 7, 48, 19, 4, 4, 50, 29, 29, 37, 17, 15, 11, 43, 8, 16, 30, 53, 7, 28, 29, 13, 48, 12, 4, 7, 7, 6, 8, 16, 8, 3, 49, 25, 50, 45, 53, 8, 13, 20, 22, 18, 33, 11, 23, 16, 13, 2, 1, 37, 8, 7, 36, 35, 20, 1, 13, 26, 13, 26, 43, 22, 37, 43, 1, 48, 33, 13, 14, 4, 15, 7, 38, 11, 27, 5, 39, 13, 23, 7, 6, 49, 41, 10, 6, 49, 15, 2, 33, 19, 30, 44, 27, 2, 28, 23, 20, 1, 15, 52, 40, 43, 14, 6, 0, 21, 9, 27, 14, 20, 37, 26, 10, 1, 12, 52, 24, 2, 24, 1, 50, 4, 26, 48, 25, 27, 48, 31, 16, 25, 24, 4, 40, 50, 1, 34, 5, 26, 15, 48, 1, 10, 9, 18, 47, 13, 25, 7, 40, 32, 2, 37, 43, 20, 52, 27, 2, 47, 6, 28, 34, 41, 25, 48, 14, 27, 34, 47, 26, 21, 50, 1, 29, 14, 45, 40, 1, 19, 13, 6, 10, 4, 1, 11, 39, 1, 1, 15, 6, 8, 41, 36, 7, 36, 26, 16, 26, 18, 12, 28, 17, 50, 3, 25, 6, 2, 41, 44, 19, 4, 43, 5, 5, 27, 38, 29, 37, 37, 28, 25, 9, 9, 23, 10, 22, 16, 13, 1, 13, 40, 13, 7, 9, 32, 11, 4, 25, 26, 40, 50, 5, 13, 9, 48, 14, 3, 34, 5, 3, 27, 2, 47, 4, 6, 13, 18, 7, 7, 4, 31, 45, 15, 29, 1, 24, 26, 37, 21, 33, 12, 16, 18, 7, 53, 2, 8, 9, 26, 14, 17, 36, 13, 29, 40, 41, 5, 21, 9, 6, 27, 43, 13, 33, 44, 48, 6, 1, 13, 12, 27, 13, 0, 38, 33, 48, 47, 21, 15, 28, 50, 52, 26, 18, 6, 47, 50, 17, 32, 11, 47, 7, 49, 1, 37, 2, 4, 21, 19, 21, 34, 11, 27, 40, 45, 36, 35, 30, 15, 1, 14, 26, 2, 20, 32, 0, 3, 32, 5, 22, 28, 20, 4, 27, 25, 29, 10, 0, 7, 7, 50, 1, 9, 53, 27, 26, 27, 4, 7, 8, 11, 11, 6, 43, 8, 38, 29, 4, 41, 34, 20, 29, 37, 7, 14, 1, 1, 21, 8, 13, 27, 27, 13, 2, 28, 0, 13, 5, 30, 27, 13, 2, 6, 13, 51, 50, 9, 4, 1, 27, 20, 38, 22, 4, 27, 12, 32, 35, 29, 20, 5, 27, 38, 30, 36, 49, 37, 16, 27, 22, 36, 6, 2, 16, 11, 11, 27, 17, 31, 37, 41, 2, 21, 33, 53, 7, 31, 33, 36, 11, 36, 14, 25, 15, 1, 19, 37, 2, 26, 7, 36, 52, 16, 6, 7, 13, 50, 43, 35, 53, 27, 39, 19, 36, 5, 7, 7, 11, 4, 35, 7, 13, 42, 29, 7, 25, 23, 36, 16, 43, 44, 1, 9, 23, 7, 24, 44, 4, 7, 11, 17, 53, 22, 48, 41, 44, 12, 50, 2, 1, 13, 1, 34, 37, 7, 8, 6, 20, 31, 25, 39, 7, 9, 44, 37, 44, 34, 44, 24, 11, 6, 12, 21, 25, 37, 15, 10, 34, 48, 23, 53, 0, 28, 5, 44, 42, 7, 39, 26, 13, 16, 32, 10, 31, 7, 43, 32, 37, 17, 48, 12, 1, 9, 10, 19, 6, 50, 3, 35, 28, 32, 21, 50, 30, 38, 9, 7, 35, 40, 26, 6, 1, 13, 3, 49, 38, 36, 6, 1, 16, 7, 1, 35, 51, 19, 1, 39, 39, 13, 26, 6, 21, 6, 36, 34, 4, 1, 33, 23, 6, 21, 17, 52, 19, 27, 43, 35, 42, 31, 20, 6, 7, 47, 45, 13, 1, 43, 7, 3, 29, 41, 50, 14, 5, 0, 6, 3, 7, 30, 3, 2, 13, 49, 7, 21, 9, 15, 1, 34, 27, 48, 40, 5, 23, 36, 4, 9, 21, 31, 4, 36, 42, 37, 32, 15, 14, 30, 4, 22, 23, 51, 27, 2, 5, 13, 1, 39, 27, 20, 4, 7, 43, 11, 26, 31, 41, 6, 19, 10, 16, 16, 22, 1, 7, 4, 20, 1, 27, 29, 11, 48, 40, 39, 50, 48, 11, 31, 27, 18, 40, 34, 22, 45, 23, 21, 6, 11, 6, 45, 1, 51, 1, 35, 4, 5, 37, 5, 50, 41, 21, 26, 24, 2, 49, 2, 13, 51, 40, 19, 27, 7, 34, 15, 1, 38, 21, 1, 24, 3, 31, 30, 16, 28, 46, 6, 7, 6, 7, 6, 34, 4, 15, 6, 13, 46, 1, 5, 34, 10, 27, 37, 9, 15, 44, 31, 4, 25, 27, 47, 38, 1, 48, 11, 20, 51, 31, 0, 21, 4, 48, 2, 6, 20, 40, 14, 44, 27, 7, 45, 46, 42, 7, 6, 36, 2, 53, 26, 49, 7, 51, 21, 6, 13, 15, 7, 34, 30, 35, 20, 50, 29, 6, 12, 37, 23, 7, 15, 21, 35, 7, 1, 27, 45, 8, 35, 33, 24, 34, 52, 6, 4, 14, 19, 13, 34, 37, 14, 41, 12, 53, 11, 40, 4, 50, 1, 0, 40, 37, 15, 38, 5, 1, 37, 38, 11, 2, 34, 6, 7, 16, 11, 36, 27, 52, 10, 36, 7, 12, 4, 37, 15, 34, 43, 20, 3, 48, 35, 5, 13, 25, 4, 29, 11, 21, 7, 6, 4, 25, 2, 0, 27, 20, 8, 11, 44, 53, 29, 37, 19, 42, 26, 46, 0, 27, 0, 19, 26, 13, 31, 50, 10, 13, 46, 7, 7, 5, 14, 21]

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
    """ return all the subject ids in a question

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


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, output):
        """
        The given outputs contain whether the question is answered correctly.
        Given the question clusters we have, find the incluster similarity in question answers
        """

        loss = 0
        sets = set(q_clusters_15subjects)
        for cluster in sets:
            #get the questions in the cluster
            questions = np.array(q_clusters_15subjects)
            questions = questions * np.where(questions == cluster , 1, 0)
            #get the answers to the questions
            answers = output[:,questions]
            #calculate the standard deviation of the answers
            std = torch.std(answers, dim=1)
            loss+= std
        return loss





class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

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

            cont_loss = ContrastiveLoss();
            loss = torch.sum((output - target) ** 2.) + lamb * model.get_weight_norm() + cont_loss.forward(output) #added the weight regularizer,. 
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        train_loss_arr.append(train_loss)
        valid_acc_arr.append(valid_acc)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    plot_loss(train_loss_arr, valid_acc_arr)

    
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
    path = os.path.join("data", "train_sparse.npz")


    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k = 50
    

    # Set optimization hyperparameters.
    lr = 0.05
    num_epoch = 15
    lamb = 0.001

    #leaving these commented out to show the process of choosing hyperparameters

    # for k in [50, 100, 200, 500]:
    #     model = AutoEncoder(train_matrix.shape[1], k)
    #     print()
    #     print("k = {}".format(k))
    #     train(model, lr, lamb, train_matrix, zero_train_matrix,
    #       valid_data, num_epoch)
    #     test_acc = evaluate(model, zero_train_matrix, test_data)
    #     print("Test Accuracy: {}".format(test_acc))

    #choosing k = 100 based on validation accuracy of 0.657, testacc 0.6627
    # for lamb in [0.001, 0.01, 0.1, 1]:
    #     model = AutoEncoder(zero_train_matrix.shape[1], 100)
    #     train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, k)
    #     test_acc = evaluate(model, zero_train_matrix, test_data)
    #     print("Test Accuracy: {}".format(test_acc))
    #the model performs significantly worse in terms of loss, less but still worse in terms of validation accuracy
    model = AutoEncoder(train_matrix.shape[1], k)
    train(model, lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch)
    test_acc = evaluate(model, zero_train_matrix, test_data)
    print("Test Accuracy: {}".format(test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
