import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=1) 
# pca = PCA(n_components=0.95) # 保留95%的方差，结果也是154个特征
test = np.array([[-4, -6],[3, 5]])
X_reduced = pca.fit_transform(test)	# 压缩
print(X_reduced)
X_recovered = pca.inverse_transform(X_reduced)
print(X_recovered)
print(pca.components_[0])
#print(pca.components_[1])
