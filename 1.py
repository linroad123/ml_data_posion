import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white")

# Sample generation
# The generated data set contains 200 samples and 2 features
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, weights=[.5, .5], random_state=17)

# Use the first 100 samples to train the model
# last 100 samples are used to visualize whether the model is well trained
# MLP is a simple feedforward neural network that can create nonlinear decision boundaries.
clf = MLPClassifier(max_iter=600, random_state=123).fit(X[:100], y[:100])

# visualize the decision function of the model
# Create a two-dimensional grid of points in the input space 
# (X and y are between -3 and 3, and the interval between each adjacent point is 0.01)
# extract the predicted probability of each point in this grid.
xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

# Generate a contour map and cover the test set
f, ax = plt.subplots(figsize=(12, 9))
contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                      vmin=0, vmax=1)
ax_c = f.colorbar(contour)
ax_c.set_label("$P(y = 1)$")
ax_c.set_ticks([0, .25, .5, .75, 1])

ax.scatter(X[100:,0], X[100:, 1], c=y[100:], s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.set(aspect="equal",
       xlim=(-3, 3), ylim=(-3, 3))
plt.show()