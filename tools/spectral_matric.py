import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2

from spectral_metric.estimator import CumulativeGradientEstimator
from spectral_metric.visualize import make_graph

labels = [
    "real",
    "exp1",
    "exp2",
    "exp3",
    "exp4",
    "perlin_freq8",
    "exp6",
    "exp7",
    "exp8",
    "1_single",
    "2_multi",
]
for i, label in enumerate(labels):
    if i == 0:
        embedding = np.loadtxt(f"./outputs/test/patent_real/{label}.csv", delimiter=",")
        target = [i for e in range(len(embedding))]
    else:
        _embedding = np.loadtxt(
            f"./outputs/test/patent_real/{label}.csv", delimiter=","
        )
        embedding = np.concatenate([embedding, _embedding])

        _target = [i for e in range(len(_embedding))]
        target = np.concatenate([target, _target])

# embedding = np.loadtxt("./data/embedding.csv", delimiter=",")
# target = np.loadtxt("./data/target.csv", delimiter=",", dtype=int)


data, target = embedding, target  # X_train, y_train
# WARNINGS: data may need to be normalized and/or rescaled
k_nearest = 3  # neigborhood size
M_sample = 1000  # Number of estimation per class
estimator = CumulativeGradientEstimator(M_sample, k_nearest)
estimator.fit(data, target)

print(estimator.csg)
print(estimator.evals)
print(estimator.evecs)

# To plot the graph
make_graph(estimator.difference, title="result", classes=labels)
