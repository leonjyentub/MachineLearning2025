from sklearn.linear_model import Ridge
import numpy as np
'''
n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
clf = Ridge(alpha=1.0)
clf.fit(X, y)

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(9).reshape(3, 3)
print(X)
poly = PolynomialFeatures(2)
y = poly.fit_transform(X)
print(y)
'''
import numpy as np
X = np.array([[1., 0., 1], [2., 1., 3], [0., 0., 1],[1,1,1]])
y = np.array([[0], [1], [2], [3]])

from sklearn.utils import shuffle
X, y = shuffle(X, y)
print(X, y)
