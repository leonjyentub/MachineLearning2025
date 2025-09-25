import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

X=np.array([
    [1,2,3,4],
    [11,12,13,14],
    [21,22,23,24],
    [31,32,33,34],
    [41,42,43,44],
    [51,52,53,54],
    [61,62,63,64],
    [71,72,73,74]
])

y=np.array([1,1,0,0,1,1,0,0])

floder = KFold(n_splits=4)
for train, test in floder.split(X,y):
    print('Train: %s | test: %s' % (train, test))
print()

sfolder = StratifiedKFold(n_splits=4)
for train, test in sfolder.split(X,y):
    print('Train: %s | test: %s' % (train, test))
 

