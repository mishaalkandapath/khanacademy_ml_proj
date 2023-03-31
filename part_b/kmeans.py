from concurrent.futures import process
from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import kneed

subjects = []
#clusters of the different subjects stored globally for visualization
clusters1 = """ 
[13 27  6 29  4 20  3 23 12 29  9  3 26  0 25 29  3 11  1 10  9  4 10  3
 16 25  2 21  9  8 17 10  1  4  2 24 16 23  1 28 15 17 22  2  1 25 28 29
  9 29  9  0  3 26 17  4 15 10 19 25 16 26 16 16 29 29 25 28 18 10 29 24
  3 10 12  4  2 23 14 23  4 17  3 23 23  4 11 28 25 10 29 25 24  8 29 11
 17 11 25 16 11 18  1 12 14 24 14  2  4 10 24 16  9 21  1 21 23  8  8 15
 29 15 15 28 29 17 10 21 14  4 26  4  4  4 24 16 16 29  9 17 18 18 17  2
 16 28 18 17 18 18 18 17 20  3 21 21  2  4 14 10  4 11  4 10 29 21 18 21
 16 14 14 29 29 18 14 17 29  8 23 23 23 29 23 21  9 17  0 29  2  4 10 21
 10  9 10 14 21 28 12  2 24 29  9 10  9 21 23  4 24  4  2  4 23 21  4  9
  2  2 19  4 19 16 21  9  9  4 18 28  5 29 16 26  7 29 17  1  4 10  3  4
 29  9 23 29 21 14 29 23 14 14 14 17 28 23 26 10 21  6  4 17 17  9 16 24
 23  9  9  9 23 23 23 17 23 14 25 25 10 10 10 16  9 29 16 17 18  4 18 18
 16 24 14 29 23  4  1  2 23  9 18 17  2  3 14 29 23 23 14 14 18 17 14  0
 25 13 27  4 20  3 23 12 29  3 26  0 25  3 11 10  3 21  8 17 16 28 22 25
 29 29  9 19 29 29 25 28 18  4 11 11 17 11 25 16 18  1 12 14 24  2 15 15
 28 12  4  9 16  7  3 23 14 21  8 21  4 24 26  0 21 24 26  9 29 23 24 29
 21 21 10 21]
"""
clusters2 = """
[ 6  3 10  8  8  5  6 11 12  8  2 11 13  4  1  8  6  0  2  2  1  1  2  1
 12  1  5  6  1 13  2  1  2  1  8 12 12 11  2 11  7  2  0  5  2  3 11  8
  2 12  1  7 11  3  2  8 10  5  3  3 12  0 12 12  1  8  1  8  6  8  8  7
 11  8  1  1  8 11  1 11  8  2 11  5  5  8  3  8  1  2 12 11 12 13  3  3
  2  3  6 12 11  6  2  1  2  7  1  8  8  8  7 12  2  2  2  6 11 11 13  5
  8  5  8  3  8  2  1  6  2  8 13  8  8  8  7 12 12  8  1  6  6  1  6 11
 12  5  1  6  6  1  6  6  5 11  6  6  8  8  2  6  1  3  8  8  8 11  6  6
 12  1  6  1  8  1  6  6  8 13 11 11 11 11 11  6  2  6  8  8  8  8  2  6
  8  1  2  6  6  8  1  8  7 10  2  8  2  5 11  8  7  8  8  1  5  6  7  1
  8  8  3  8  3 12  2  2  2  8  8  3 14 11 12  2  9 12  1  2  8  8  1  8
 11  2 11  8  2  1 11 11  1  1  1  6  3 11  3  1  5 10  1  6  6  2 12  7
 11  2  2  2 11 11 11  2 11  1  1  1  7  2  8 12  2  2 12  2  8  1  1  1
 12  7  1  8 11 12  2  5  5  2  1  2  8  1  6 12  5 11  1  6  1  6  1  7
  1  6  3  8  5  6 11 12  8 11 13  4  1  6  0  2  1  6 13  2 12 11  0  3
  8 12  1  3  1  8  1  8  6  8  3  3  2  3  6 12  6  2  1  2  7  8  5  5
  3  1  7  1 12  9  1 11  1  6 13  5  8  7 13  8  5  7 11  2 11 11  8  1
  2  5  2  5]
"""

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


def process_chat(all_string):
    # got a list of all lines, with words in them seperated too:
    m = Word2Vec(sentences=all_string, vector_size=50, min_count=1, sg=1)
    l = []
    for line in all_string:
        l.append(vectorizer(line, m))

    X = np.array(l)
    wcss = []
    models = []
    for i in range(30, 31):
        kmeans = KMeans(n_clusters = i, init="k-means++" , random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        models.append(kmeans)
    # plt.plot(range(2, 50), wcss)
    # plt.show()

    # kneedle = kneed.KneeLocator(x=range(2, 50), y=wcss, curve="convex", direction="decreasing", online=True, S=1)
    # knee_point = kneedle.elbow
    #print(knee_point)

    # main_model = models[knee_point - 2]
    main_model = models[0]
    labels = main_model.fit_predict(X)
    # label_dict = {}
    # for idx, label in enumerate(labels):
    #     label_dict[label] = label_dict.get(label, "") + " ".join(lines[idx])#label_dict.get(label, []) + [ " ".join(lines[idx])]
    return labels

def vectorizer(sent, m):
    vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                vec = m.wv[w]
            else:
                vec = np.add(vec, m.wv[w])
            numw+=1
        except:
            pass
    return np.asarray(vec)/numw

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

if __name__ == "__main__":
    load_subjects()
    groups = process_chat(subjects)
    q_meta_data = question_subject_metadata()
    #cluster the questions:
    vectorized_matrix = np.ndarray((len(q_meta_data), 30)) #rows represent questions, columns the number of occurences of a subject in cluster i
    for question in range(len(q_meta_data)):
        qs = q_meta_data[question] #get the subjects in this question
        qs = groups[qs] #an array of indices
        #make the question row of vectorized_matrix using qs
        for i in range(30):
            vectorized_matrix[question][i] = np.count_nonzero(qs == i)
    
    #cluster this vectorized matrix now: 
    wcss = []
    models = []
    for i in range(2, 100):
        kmeans = KMeans(n_clusters = i, init="k-means++" , random_state=42)
        kmeans.fit(vectorized_matrix)
        wcss.append(kmeans.inertia_)
        models.append(kmeans)
    plt.plot(range(2, 100), wcss)
    plt.show()

    kneedle = kneed.KneeLocator(x=range(2, 100), y=wcss, curve="convex", direction="decreasing", online=True, S=1)
    knee_point = kneedle.elbow

    main_model = models[knee_point - 2]
    labels = main_model.fit_predict(vectorized_matrix)
    print(labels.tolist())