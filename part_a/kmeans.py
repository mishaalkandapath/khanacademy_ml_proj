from concurrent.futures import process
from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import kneed

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


def process_chat(all_string):
    # got a list of all lines, with words in them seperated too:
    m = Word2Vec(sentences=all_string, vector_size=50, min_count=1, sg=1)
    l = []
    for line in all_string:
        l.append(vectorizer(line, m))

    X = np.array(l)
    wcss = []
    models = []
    for i in range(15, 16):
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

if __name__ == "__main__":
    load_subjects()
    groups = process_chat(subjects)
    print(len(groups))
    print(groups)