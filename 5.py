import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white")
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, weights=[.5, .5], random_state=17)
clf = MLPClassifier(max_iter=600, random_state=123).fit(X[:100], y[:100])
xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)


def plot_decision_boundary(X_orig, y_orig, probs_orig, chaff_X=None, chaff_y=None, probs_poisoned=None):
    f, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(X_orig[100:,0], X_orig[100:, 1], c=y_orig[100:], s=50,
            cmap="gray", 
            edgecolor="black", linewidth=1)

    if all([(chaff_X is not None),
            (chaff_y is not None),
            (probs_poisoned is not None)]):
        ax.scatter(chaff_X[:,0], chaff_X[:, 1], 
                   c=chaff_y, s=50, cmap="gray", 
                   marker="*", edgecolor="black", linewidth=1)
        ax.contour(xx, yy, probs_orig, levels=[.5], 
                   cmap="gray", vmin=0, vmax=.8)
        ax.contour(xx, yy, probs_poisoned, levels=[.5], 
                   cmap="gray")
    else:
        ax.contour(xx, yy, probs_orig, levels=[.5], cmap="gray")

    ax.set(aspect="equal", xlim=(-3, 3),ylim=(-3, 3))
    plt.show()
num_chaff = 5
chaff_X = np.array([np.linspace(-2,-1,num_chaff), np.linspace(0.1,0.1, num_chaff)]).T
chaff_y = np.ones(num_chaff)
clf.partial_fit(chaff_X,chaff_y)
clf.partial_fit(chaff_X,chaff_y)
clf.partial_fit(chaff_X,chaff_y)
clf.partial_fit(chaff_X,chaff_y)
clf.partial_fit(chaff_X,chaff_y)
clf.partial_fit(chaff_X,chaff_y)
clf.partial_fit(chaff_X,chaff_y)
clf.partial_fit(chaff_X,chaff_y)
clf.partial_fit(chaff_X,chaff_y)
clf.partial_fit(chaff_X,chaff_y)
clf.partial_fit(chaff_X,chaff_y)
clf.partial_fit(chaff_X,chaff_y)
clf.partial_fit(chaff_X,chaff_y)
clf.partial_fit(chaff_X,chaff_y)
clf.partial_fit(chaff_X,chaff_y)
probs_poisoned = clf.predict_proba(grid)[:,1].reshape(xx.shape)
plot_decision_boundary(X,y,probs,chaff_X,chaff_y,probs_poisoned)